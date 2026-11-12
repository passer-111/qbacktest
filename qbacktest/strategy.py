# -*- coding: utf-8 -*-
"""策略接口：框架与策略之间唯一的契约。

框架负责所有"与选股无关"的事情——数据加载、因子面板（Beta/波动率/MACD）、
KeyIndex 财务/分红检索、撮合、费用、净值、进度上报、绩效统计。
策略只需要实现一个方法：

    def select(self, ctx) -> (list[str], str)

返回「目标持仓代码列表」和「风格标签」（标签只用于前端展示）。
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Ctx:
    """每个调仓日传给策略的只读快照。全部是 numpy/pandas 结构，不做拷贝。"""

    date: object            # 决策日（EXEC_LAG=1 时是前一交易日）
    row: int                # 决策日在日期轴上的行号

    P: np.ndarray           # 收盘价（框架口径：前复权）
    PR: np.ndarray          # 收盘价（不复权真实价，估值类因子必须用它）
    V: np.ndarray           # 成交量
    O: np.ndarray           # 开盘价（撮合用）

    beta_v: np.ndarray      # Beta 面板（对基准，rolling 窗口见 cfg）
    vol_v: np.ndarray       # 波动率面板（同窗口的年化标准差）
    fin_idx: dict           # FN 字段 -> KeyIndex（支持 .latest(yyyymmdd)）
    div_idx: object         # 分红 KeyIndex（支持 .window_sum(lo, hi)）
    first_idx: np.ndarray   # 每只股票在价格矩阵里的首个有效行（判上市天数）
    code_pos: dict          # code -> 列号
    codes: list             # 列号 -> code

    hist: object            # 基准的月线 MACD 柱值（pd.Series，可能为 None）
    hold: set               # 当前持仓代码集合
    cfg: dict               # 运行配置（回测区间、资金、费用、窗口等）


class Strategy:
    """策略基类。子类只需覆盖 name 和 select。"""

    name = 'base'

    def select(self, ctx: Ctx):
        raise NotImplementedError('策略必须实现 select(ctx)')

    # 可选钩子：换仓后做一次记录（比如打印用了哪些过滤条件）
    def on_rebalance(self, ctx: Ctx, target, style):
        pass


# ---------------- 策略里常用的工具函数 ----------------

def latest(fin_idx, field, asof_int, n_codes):
    """取「asof 之前最新一期」的某个财务字段；字段缺失时返回全 NaN。"""
    idx = fin_idx.get(field)
    if idx is None:
        return np.full(n_codes, np.nan)
    return idx.latest(asof_int)


def safe_div(a, b):
    """带零保护的除法，返回 NaN 而不是 inf。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(np.abs(b) > 1e-12, a / b, np.nan)
    return np.where(np.isfinite(out), out, np.nan)
