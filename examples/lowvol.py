# -*- coding: utf-8 -*-
"""示例策略：低波动 + 大市值。

存在的意义是演示怎么写一个策略插件，不是推荐这个策略。
它只做三件事：
  1. 剔除 ST / 停牌 / 上市不足一年 / 市值后 30%
  2. 按 120 日波动率升序，取最低的 20 只
  3. 等权持有，每次调仓重新选

看这个文件就知道框架给了策略什么、策略该返回什么。
"""
import numpy as np

from qbacktest.strategy import Strategy, Ctx, latest, safe_div


class LowVolLargeCap(Strategy):
    name = 'lowvol_largecap'

    # 策略参数：框架不关心，自己维护即可
    TOP_N = 20
    MIN_MV_PCT = 0.30      # 剔除总市值排名后 30%
    MAX_VOL = 0.60         # 年化波动率上限

    def select(self, ctx: Ctx):
        cfg = ctx.cfg
        price = ctx.P[ctx.row]
        praw = ctx.PR[ctx.row] if ctx.PR is not None else price

        alive = np.isfinite(price) & (price > 0)
        listed = (ctx.row - ctx.first_idx) >= cfg.get('min_list_days', 250)
        tradable = np.isfinite(ctx.V[ctx.row]) & (ctx.V[ctx.row] > 0)
        pool = alive & listed & tradable
        if pool.sum() < 50:
            return [], 'empty'

        asof = int(ctx.date.strftime('%Y%m%d'))

        # 总市值 = 不复权价 × 总股本（必须用真实价，前复权价会把市值算小）
        cap = latest(ctx.fin_idx, 'FN238', asof, len(ctx.codes))
        mv = safe_div(praw, 1.0) * cap
        mvok = np.isfinite(mv) & (mv > 0)
        if mvok.sum() > 200:
            thr = np.nanquantile(mv[mvok], self.MIN_MV_PCT)
            pool &= mvok & (mv >= thr)

        # 波动率面板（框架已按窗口算好）
        vol = ctx.vol_v[ctx.row]
        ok = pool & np.isfinite(vol) & (vol > 0) & (vol < self.MAX_VOL)
        idx = np.flatnonzero(ok)
        if len(idx) < 5:
            return [], 'empty'

        order = idx[np.argsort(vol[idx])][:self.TOP_N]
        return [ctx.codes[i] for i in order], 'lowvol'
