# -*- coding: utf-8 -*-
"""向量化回测引擎。

为速度做了三处前置计算：
  1. Beta：ret.rolling(120).cov(bench) 一次算出全历史面板，不再每月逐只循环
  2. 财务/分红：排序 + 全局 searchsorted，O(log n) 查「asof 之前最新一期」
  3. 选股：numpy 掩码 + argsort，不再用 pandas 多重布尔索引

用法:
    python backtest_fast.py            # 前复权（聚宽/QMT 口径）
    python backtest_fast.py --raw      # 不复权 + 除权日现金分红入账
"""
import os
import sys
import time
from collections import OrderedDict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

from . import progress
from .strategy import Ctx, Strategy

# 由调用方注入具体策略；框架本身不含任何选股逻辑
STRATEGY = None
CONFIG = {}

# 数据目录：默认与框架同级的上两层 localdata/，可用 QBACKTEST_DATA 覆盖
BASE = os.path.abspath(os.environ.get(
    'QBACKTEST_DATA',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'localdata')))
BT_START = '2021-11-08'
BT_END = '2026-11-06'
CAPITAL = 1_000_000.0
TARGET_NUM = 6
BUFFER = 1.7
COMMISSION = 0.00015
STAMP_TAX = 0.001
MIN_LIST_DAYS = 250
BETA_WIN = 120
BENCH = '000300.SH'
ADJ = '--raw' not in sys.argv
#   adj = 前复权价（与聚宽 196.69% 的结果吻合）
#   raw = 不复权真实价（口径更"干净"，但会偏离聚宽）
SAVE = True          # 扫描时关掉，不覆盖正式结果文件
# 撮合时点：1 = 用前一根 bar 的数据选股、当日开盘成交（聚宽口径，无未来函数）
#           0 = 用当日收盘数据选股、当日开盘成交（旧版，含未来数据）
EXEC_LAG = 1
# Beta 合理区间：实测全市场有 3.6% 的股票算出负 Beta（极端的如冀中能源 -6.17、
# 兖矿 -5.90，是停牌/跳价导致的协方差失真）。按 Beta 升序选股时这些垃圾值
# 会顶在 0.2~0.3 的银行前面，整条低波线被污染。所以卡一个物理区间。

# ---- 红利质量硬门槛（对齐中证红利低波 / 浙商研报 / 景顺"假红利四分类"）----


# ---- 选样空间约束（目标已从"复刻红利"改为"高收益低回撤"，这些是硬筛子）----

for _a in sys.argv:
    if _a.startswith('--exec='):
        EXEC_LAG = 0 if _a.split('=', 1)[1] == 'same' else 1

CACHE = os.path.join(BASE, 'cache_bars_adj.pkl' if ADJ else 'cache_bars_raw.pkl')


_MEM = {}          # 进程内缓存：参数扫描时只读一次盘


# ---------------------------------------------------------------- 数据
def load_bars():
    if 'bars' in _MEM:
        return _MEM['bars']
    sub = 'daily' if ADJ else 'daily_raw'
    stamp = max(os.path.getmtime(os.path.join(BASE, sub, f))
                for f in os.listdir(os.path.join(BASE, sub)))
    if os.path.exists(CACHE):
        try:
            obj = pd.read_pickle(CACHE)
            if obj.get('sub') == sub and obj.get('stamp') == stamp:
                _MEM['bars'] = obj
                return obj
        except Exception:
            pass

    d = os.path.join(BASE, sub)
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.parquet'))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(['date', 'code']).sort_values(['date', 'code'])

    px = df.pivot(index='date', columns='code', values='close')
    op = df.pivot(index='date', columns='code', values='open')
    vol = df.pivot(index='date', columns='code', values='volume')
    obj = {'px': px, 'op': op, 'vol': vol, 'sub': sub, 'stamp': stamp}
    try:
        pd.to_pickle(obj, CACHE)
    except Exception:
        pass
    _MEM['bars'] = obj
    return obj


RAW_CACHE = os.path.join(BASE, 'cache_raw_close.pkl')


def load_raw_close():
    """不复权收盘价（只取估值用），带缓存。"""
    d = os.path.join(BASE, 'daily_raw')
    stamp = max(os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d))
    if os.path.exists(RAW_CACHE):
        try:
            obj = pd.read_pickle(RAW_CACHE)
            if obj.get('stamp') == stamp:
                return obj['px']
        except Exception:
            pass
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.parquet'))
    df = pd.concat([pd.read_parquet(f, columns=['date', 'code', 'close'])
                    for f in files], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(['date', 'code'])
    px = df.pivot(index='date', columns='code', values='close')
    try:
        pd.to_pickle({'px': px, 'stamp': stamp}, RAW_CACHE)
    except Exception:
        pass
    return px


def load_fin():
    d = os.path.join(BASE, 'fin')
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.parquet'))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['announce_time'] = pd.to_datetime(df['announce_time'].astype(str), errors='coerce')
    df = df.dropna(subset=['announce_time'])
    return df.sort_values(['code', 'announce_time'])


def load_div():
    d = os.path.join(BASE, 'div')
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.parquet'))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = pd.DataFrame({
        'code': df['code'].values,
        'ex_date': pd.to_datetime(df['Date']).values,
        'dps': pd.to_numeric(df.get('Bonus'), errors='coerce').values / 10.0,
    }).dropna(subset=['ex_date'])
    return out.sort_values(['code', 'ex_date'])


class KeyIndex(object):
    """把 (code, date, value) 排成全局键，支持「某代码 asof 之前的最新一条」O(log n) 查询。"""

    BIG = 10 ** 10

    def __init__(self, codes_list, dates_series, values, code_pos):
        add = pd.to_datetime(pd.Series(dates_series)).dt.strftime('%Y%m%d').astype('int64').values
        cid = np.array([code_pos.get(c, -1) for c in codes_list], dtype='int64')
        keep = cid >= 0
        self.keys = cid[keep] * self.BIG + add[keep]
        self.vals = values[keep] if values is not None else None
        order = np.argsort(self.keys, kind='stable')
        self.keys = self.keys[order]
        self.vals = self.vals[order]
        # 每个 code 的起始位置（用于判断「该代码有没有数据」）
        self.cid_sorted = self.keys // self.BIG
        self.starts = np.searchsorted(self.cid_sorted, np.arange(len(code_pos)), 'left')
        self.ends = np.searchsorted(self.cid_sorted, np.arange(len(code_pos)), 'right')
        if self.vals is not None:
            self.cum = np.concatenate([[0.0], np.cumsum(self.vals)])

    def latest(self, asof_int):
        q = np.arange(len(self.starts), dtype='int64') * self.BIG + asof_int
        pos = np.searchsorted(self.keys, q, 'right') - 1
        ok = (pos >= self.starts) & (pos < self.ends)
        out = np.full(len(self.starts), np.nan)
        out[ok] = self.vals[pos[ok]]
        return out

    def window_sum(self, lo_int, hi_int):
        q_hi = np.arange(len(self.starts), dtype='int64') * self.BIG + hi_int
        q_lo = np.arange(len(self.starts), dtype='int64') * self.BIG + lo_int
        hi = np.searchsorted(self.keys, q_hi, 'right')
        lo = np.searchsorted(self.keys, q_lo, 'right')
        return self.cum[hi] - self.cum[lo]


def build_fin_index(fin, code_pos):
    """把财务表按字段拆成多个 KeyIndex，只在调仓日查一次"""
    codes = fin['code'].tolist()
    fields = ['FN1', 'FN8', 'FN40', 'FN74', 'FN86', 'FN96',
              'FN107', 'FN183', 'FN184', 'FN189', 'FN238']
    out = {}
    for f in fields:
        if f not in fin.columns:
            out[f] = None
            continue
        out[f] = KeyIndex(codes, fin['announce_time'],
                          pd.to_numeric(fin[f], errors='coerce').values, code_pos)
    return out


def build_div_index(div, code_pos):
    codes = div['code'].tolist()
    return KeyIndex(codes, div['ex_date'], div['dps'].values, code_pos)


# ---------------------------------------------------------------- 因子
def month_first_days(dates):
    s = pd.Series(dates)
    g = s.groupby([s.dt.year, s.dt.month])
    return set(g.min())


_VOL = {}


def vol_panel(px):
    """120 日收益率标准差面板（真实波动率，不是 Beta）。

    聚宽那份持仓是「工行/农行/中行/交行/成都银行/皖通高速/神华」——
    特征是大市值、低波动、央国企。用真实波动率排序比用 Beta 更接近这种偏好。
    """
    if 'v' in _VOL:
        return _VOL['v']
    ret = px.pct_change()
    out = ret.rolling(BETA_WIN, min_periods=max(20, BETA_WIN // 2)).std()
    _VOL['v'] = out
    return out


_BETA = {}


def beta_panel(px):
    """一次算出全历史 Beta 面板（对沪深300，BETA_WIN 日窗口）。按窗口缓存，扫描时复用。"""
    if BETA_WIN in _BETA:
        return _BETA[BETA_WIN]
    ret = px.pct_change()
    if BENCH not in ret.columns:
        return None
    br = ret[BENCH]
    mp = max(20, BETA_WIN // 2)
    rm = ret.rolling(BETA_WIN, min_periods=mp).mean()
    bm = br.rolling(BETA_WIN, min_periods=mp).mean()
    cov = ret.mul(br, axis=0).rolling(BETA_WIN, min_periods=mp).mean() - rm.mul(bm, axis=0)
    var = br.rolling(BETA_WIN, min_periods=mp).var()
    out = cov.div(var, axis=0)
    _BETA[BETA_WIN] = out
    return out


def macd_monthly(bench):
    m = bench.resample('ME').last().dropna()
    dif = m.ewm(span=12, adjust=False).mean() - m.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    return (dif - dea) * 2


# ---------------------------------------------------------------- 主流程
def run():
    progress.start(1, params={'口径': '前复权' if ADJ else '不复权+分红',
                              '目标只数': TARGET_NUM, '缓冲': BUFFER,
                              'Beta窗口': BETA_WIN, '区间': '%s~%s' % (BT_START, BT_END)})
    t0 = time.time()
    progress.phase('加载行情数据')
    bars = load_bars()
    px, op, vol = bars['px'], bars['op'], bars['vol']
    # 估值类因子必须用「不复权真实价」：前复权价把历史价格压低了，
    raw = load_raw_close().reindex(index=px.index, columns=px.columns)
    progress.phase('加载财务与分红')
    fin = load_fin()
    div = load_div()
    print('加载 %.1fs ｜ 日线 %s~%s %d 只 ｜ 财务 %d 行 ｜ 分红 %d 条'
          % (time.time() - t0, px.index.min().date(), px.index.max().date(),
             px.shape[1], len(fin), len(div)))

    codes = list(px.columns)
    code_pos = {c: i for i, c in enumerate(codes)}

    P = px.values.astype('float64')
    PR = raw.values.astype('float64') if raw is not None else None
    O = op.reindex(columns=codes).values.astype('float64')
    V = vol.reindex(columns=codes).values.astype('float64')
    dates = px.index

    P0 = np.isfinite(P)
    first_idx = np.argmax(P0, axis=0)

    progress.phase('计算 Beta 面板')
    t0 = time.time()
    beta = beta_panel(px)
    beta_v = beta.reindex(columns=codes).values if beta is not None else None
    vol_v = vol_panel(px).reindex(columns=codes).values
    print('Beta 面板 %.1fs' % (time.time() - t0))

    progress.phase('建财务/分红索引')
    t0 = time.time()
    fin_idx = build_fin_index(fin, code_pos)
    div_idx = build_div_index(div, code_pos)
    print('财务/分红索引 %.1fs' % (time.time() - t0))

    hist = macd_monthly(px[BENCH]) if BENCH in px.columns else None

    bt_mask = (dates >= BT_START) & (dates <= BT_END)
    bt_rows = np.flatnonzero(bt_mask)
    bt_dates = dates[bt_rows]
    rebal = month_first_days(bt_dates)
    total_reb = len(rebal)
    progress.start(total_reb, params={
        '价格口径': '前复权' if ADJ else '不复权+分红',
        '策略': getattr(STRATEGY, 'name', type(STRATEGY).__name__),
        '区间': '%s ~ %s' % (BT_START, BT_END)})

    # 分红入账表（--raw 模式）
    div_map = {}
    if not ADJ:
        for c, ex, dps in zip(div['code'], div['ex_date'], div['dps']):
            if dps and dps > 0:
                div_map.setdefault(pd.Timestamp(ex), []).append((c, float(dps)))

    holding = OrderedDict()
    cash = CAPITAL
    nav_rows, trades, picks = [], [], []
    date2row = {d: i for i, d in enumerate(dates)}

    for d in bt_dates:
        row = date2row[d]
        for c, dps in div_map.get(pd.Timestamp(d), []):
            if c in holding and holding[c] > 0:
                cash += holding[c] * dps

        is_reb = d in rebal
        if is_reb:
            # 撮合时点对齐：聚宽是「用 T-1 及之前的数据决策，T 日开盘市价成交」。
            # 原版拿当日收盘价（P[row]）选股、又按当日开盘价（O[row]）成交，
            # 等于用今天的收盘价去今天的开盘价买——用了未来数据，收益虚高。
            # EXEC_LAG=1 时改成：选股用前一根 bar，成交仍在当日开盘。
            srow = row - EXEC_LAG if row - EXEC_LAG >= 0 else row
            sdate = dates[srow]
            ctx = Ctx(date=sdate, row=srow, P=P, PR=PR, V=V, O=O,
                      beta_v=beta_v, vol_v=vol_v, fin_idx=fin_idx,
                      div_idx=div_idx, first_idx=first_idx,
                      code_pos=code_pos, codes=codes, hist=hist,
                      hold=set(c for c, s in holding.items() if s > 0),
                      cfg=CONFIG)
            target, style = STRATEGY.select(ctx)
            picks.append((str(d.date())[:10], ','.join(target), style))
            holding, cash = rebalance(row, target, holding, cash, O, code_pos, trades, d)

        mv = 0.0
        prow = P[row]
        for c, sh in holding.items():
            p = prow[code_pos[c]]
            if np.isfinite(p):
                mv += sh * p
        nav_rows.append((d, cash + mv, cash, len(holding)))

        if is_reb:
            progress.step(len(picks), total_reb, str(d.date())[:10],
                          nav_point=(str(d.date())[:10], (cash + mv) / CAPITAL),
                          pick=(str(d.date())[:10], ','.join(target), style),
                          trade=(trades[-1] if trades else None))

    if SAVE:
        pd.DataFrame(picks, columns=['date', 'codes', 'style']).to_parquet(
            os.path.join(BASE, 'picks.parquet'), index=False)
    nav = pd.DataFrame(nav_rows, columns=['date', 'nav', 'cash', 'n_hold']).set_index('date')
    return nav, trades


def rebalance(row, target, holding, cash, O, code_pos, trades, d):
    o = O[row]

    # 成分不变就不动：聚宽版只在「股票进出目标池」时交易，
    # 不做每月等权再平衡。原来每月剃一遍会多出近 200 笔无效交易
    # （每笔都交佣金+印花税），这也是回撤比聚宽大 3pp 的原因之一。
    cur_names = set(c for c, sh in holding.items() if sh > 0)
    if cur_names == set(target):
        return holding, cash

    for c in list(holding.keys()):
        if c in target:
            continue
        p = o[code_pos[c]]
        if not np.isfinite(p) or p <= 0:
            continue
        sh = holding.pop(c)
        amt = sh * p
        cash += amt - (max(5.0, amt * COMMISSION) + amt * STAMP_TAX)
        trades.append((d, c, 'sell', sh, p))

    if not target:
        return holding, cash

    # 只买「新进池且当前没有」的票，钱平摊给它们；已持有的不加不减
    missing = [c for c in target if holding.get(c, 0) <= 0]
    if not missing:
        return holding, cash
    budget = cash / len(missing)
    for c in missing:
        p = o[code_pos[c]]
        if not np.isfinite(p) or p <= 0:
            continue
        sh = int(budget * 0.98 / p / 100) * 100
        if sh <= 0:
            continue
        cost = sh * p
        fee = max(5.0, cost * COMMISSION)
        if cost + fee > cash:
            sh = int(cash * 0.97 / p / 100) * 100
            if sh <= 0:
                continue
            cost = sh * p
            fee = max(5.0, cost * COMMISSION)
        cash -= cost + fee
        holding[c] = holding.get(c, 0) + sh
        trades.append((d, c, 'buy', sh, p))
    return holding, cash


def report(nav, trades, secs):
    n = nav['nav']
    years = (n.index[-1] - n.index[0]).days / 365.25
    total = n.iloc[-1] / CAPITAL - 1
    cagr = (n.iloc[-1] / CAPITAL) ** (1 / years) - 1
    dd = (n / n.cummax() - 1).min()
    print('=' * 60)
    print('区间 %s ~ %s（%.2f 年）｜ 口径 %s'
          % (n.index[0].date(), n.index[-1].date(), years, '前复权' if ADJ else '不复权+现金分红'))
    print('期末 %.0f ｜ 累计 %.2f%% ｜ 年化 %.2f%% ｜ 最大回撤 %.2f%%'
          % (n.iloc[-1], total * 100, cagr * 100, dd * 100))
    print('交易 %d 笔 ｜ 总耗时 %.1fs' % (len(trades), secs))
    prev = CAPITAL
    for k, v in n.resample('YE').last().items():
        print('  %d: %+.1f%%' % (k.year, (v / prev - 1) * 100))
        prev = v
    return {'total_return': round(float(total) * 100, 2),
            'cagr': round(float(cagr) * 100, 2),
            'max_dd': round(float(dd) * 100, 2),
            'trades': len(trades), 'final': round(float(n.iloc[-1]), 0),
            'mode': '前复权' if ADJ else '不复权+现金分红',
            'secs': round(secs, 1)}


if __name__ == '__main__':
    t0 = time.time()
    nav, trades = run()
    secs = time.time() - t0
    nav.to_parquet(os.path.join(BASE, 'nav.parquet'))
    pd.DataFrame(trades, columns=['date', 'code', 'side', 'shares', 'price']).to_parquet(
        os.path.join(BASE, 'trades.parquet'), index=False)
    stats = report(nav, trades, secs)

    n = nav['nav']
    yearly, prev = [], CAPITAL
    for k, v in n.resample('YE').last().items():
        yearly.append([str(k.year), round(float(v / prev - 1) * 100, 2)])
        prev = v
    progress.finish(stats, yearly)