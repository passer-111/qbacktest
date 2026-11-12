# -*- coding: utf-8 -*-
"""回测入口：加载数据 → 注入策略 → 跑 → 打印绩效。

    python run.py                                  # 用 examples/lowvol.py
    python run.py --strategy examples.lowvol:LowVolLargeCap
    python run.py --start 2021-11-08 --end 2026-11-06
    python run.py --raw                            # 不复权 + 现金分红入账
    QBACKTEST_DATA=D:/mydata python run.py         # 数据目录换地方

策略用 `模块路径:类名` 指定，只要实现了 qbacktest.strategy.Strategy 即可。
"""
import argparse
import importlib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from qbacktest import engine as E


def load_strategy(spec):
    if ':' not in spec:
        spec = spec + ':Strategy'
    mod_name, cls_name = spec.split(':', 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='examples.lowvol:LowVolLargeCap')
    ap.add_argument('--start', default=E.BT_START)
    ap.add_argument('--end', default=E.BT_END)
    ap.add_argument('--capital', type=float, default=E.CAPITAL)
    ap.add_argument('--raw', action='store_true', help='不复权 + 现金分红入账')
    ap.add_argument('--data', default=None, help='数据目录（覆盖默认）')
    ap.add_argument('--bt-win', type=int, default=E.BETA_WIN, help='Beta/波动率窗口')
    ap.add_argument('--no-save', action='store_true', help='不覆盖 picks.parquet')
    args = ap.parse_args()

    if args.data:
        E.BASE = os.path.abspath(args.data)
        E.CACHE = os.path.join(E.BASE, 'cache_bars_adj.pkl' if E.ADJ else 'cache_bars_raw.pkl')

    E.BT_START, E.BT_END = args.start, args.end
    E.CAPITAL = args.capital
    E.BETA_WIN = args.bt_win
    E.ADJ = not args.raw
    if args.no_save:
        E.SAVE = False

    E.STRATEGY = load_strategy(args.strategy)
    E.CONFIG = {
        'start': args.start, 'end': args.end, 'capital': args.capital,
        'adj': E.ADJ, 'beta_win': args.bt_win,
        'min_list_days': E.MIN_LIST_DAYS,
        'commission': E.COMMISSION, 'stamp_tax': E.STAMP_TAX,
        'bench': E.BENCH, 'exec_lag': E.EXEC_LAG,
    }

    print('数据目录 %s' % E.BASE)
    print('策略 %s ｜ 区间 %s ~ %s ｜ 资金 %.0f ｜ 价格口径 %s'
          % (E.STRATEGY.name, args.start, args.end, args.capital,
             '前复权' if E.ADJ else '不复权+分红'))

    nav, trades = E.run()
    E.report(nav, trades, 0.0)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
    os.makedirs(out, exist_ok=True)
    nav.to_parquet(os.path.join(out, 'nav.parquet'))
    print('\n净值已存 %s' % os.path.join(out, 'nav.parquet'))


if __name__ == '__main__':
    main()
