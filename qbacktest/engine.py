# -*- coding: utf-8 -*-
"""向量化回测引擎。

为速度做了三处前置计算：
  1. Beta：ret.rolling(120).cov(bench) 一次算出全历史面板，不再每月逐只循环
  2. 财务/分红：排序 + 全局 searchsorted，O(log n) 查「asof 之前最新一期」
  3. 选股：numpy 掩码 + argsort，不再用 pandas 多重布尔索引

用法:
    python backtest_fast.py                        # 默认：不复权 + 红利再投资 + 扣税（实盘口径）
    python backtest_fast.py --adj                  # 前复权（税前，用于和外部基准对比）
    python backtest_fast.py --begin 2016-01-01     # 指定起点

口径定义（务必固定，别再来回改）：
    默认（ADJ=False）
        价格：不复权真实成交价（daily_raw）
        分红：除权日按当日收盘价立即买入同一只股票（CRSP 总收益口径）
        送转：股数 ×(1 + ShareBonus/10)
        配股：扣现金、股数 ×(1 + Allotment/10)
        红利税：按持股期分档 0/10/20%
        → 这是「实盘能复现」的数字

    --adj（ADJ=True）
        价格：前复权（xtdata front_ratio，实测是【税前】口径，K 中位 0.997）
        分红：已折进价格，不再扣税（否则重复扣，实测低估 0.5pp）
        → 这是「和指数/Tushare 对比」的数字

两者实测差 0.08pp（2016-2026：11.69% vs 11.61%），差异来自税前 vs 税后。
"""
import json
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

BASE = (os.environ.get('QUANT_DATA')
        or os.path.dirname(os.path.abspath(__file__)))

# 清洗后的数据目录（由 clean_l2.py 生成）：
#   clean/daily/      前复权，按板块阈值清洗过，极端值已剔除
#   clean/daily_raw/  不复权，带 amount，只做了停牌处理和上市期过滤
# 原始 daily/ 和 daily_raw/ 保留不动，随时可以 A/B 对比。
CLEAN = os.path.join(BASE, 'clean')
# 回测区间：2014-01 起（数据从 2013-01 拉，留一年做上市期与 Beta 预热）
# 只看 2021-2026 会严重高估——那 5 年是红利顺风，2019-2020 成长牛里策略跑输沪深300 54pp
BT_START = '2014-01-01'
BT_END = '2026-11-06'
# 初始资金。默认 100 万：12 只等权，单只约 8.3 万。
# 滑点模型和规模无关——它按「本次金额 ÷ 当日成交额」自适应，
# 所以 100 万时冲击成本会自然变小（实测年化比 1000 万高约 1.1pp）。
CAPITAL = 1_000_000.0
# 持仓只数。原策略固定 6 只（2 低波 + 4 价值），这里改成可调并按 1:2 比例缩放。
# 8 只在 2021-2026（红利顺风段）最优，但拉到 2014-2026 全区间后翻案：
#   4只 Calmar 0.34 ｜ 6只 0.34 ｜ 8只 0.38 ｜ 10只 0.41 ｜ 12只 0.47 ｜ 15只 0.51
#   回撤同步改善：-52% → -42.65%(12只) → -37.79%(15只)
# 取 12 只：Calmar 0.47，且训练段(2014-19) 197% / 检验段(2020+) 244% 两段都稳；
# 15 只 Calmar 更高但检验段掉到 202%，且 1163 笔换手偏多。
TARGET_NUM = 12
# 补位：两条线选不满 N 只时，从剩余候选里补齐。
# 实测是负优化，默认关闭：选 7.8 只时平均现金只占 0.9%（买入按剩余现金平摊，
# 空仓位并不闲置资金），补进来的却是两条线都没选中的次级候选，
# 摊薄收益：279.69% -> 266.90%，Calmar 1.98 -> 1.93。
FILL_SLOTS = False
BUFFER = 1.7

# ---- 交易成本（按公司实际费率，2026-11-14 用户提供）----
# 股票佣金：万 1（之前假设万 1.5）
# 过户费：  万 0.1（沪深双边，2022 年起深市也收）
# 印花税：  千 1（仅卖出）
# 分红税：  差别化，持股 >1 年免（见 DIV_TAX）
#
# 一次完整换手 = (万1 + 万0.1) × 2 + 千1 = 0.122%
# 月调仓 12 次/年 → 年成本 1.46%（万1.5 时是 1.58%，差 0.12pp）
COMMISSION = 0.0001
TRANSFER_FEE = 0.00001      # 过户费万 0.1，和佣金一起计入「买入/卖出」成本

# ---- 滑点模型（按 A 股实际盘口校准）----
# 两部分相加：
#   1) 价差成本 = 吃 1 个 tick（A 股主板最小变动 0.01 元）占股价的比例
#      这就是为什么低价股滑点占比高：3 元股 1 tick = 33bp，30 元股只有 3.3bp
#      实际挂单密集时常只吃半个 tick，所以下限设 1bp、上限 30bp
#   2) 冲击成本 = 150bp × sqrt(本次金额 ÷ 当日成交额)
#      平方根模型，150bp 对应日波动率约 1.5%（红利股的实际水平）
# 参考口径：QMT/PTrade 给日线策略的建议是单边 5~10bp（对应 20~40 元的高价股），
#          中盘股 10~20bp；我们持仓股价多在 5~30 元，算出来正好落在这个区间。
SLIP_ON = True
SLIP_TICK = 0.01         # A 股最小变动价位（元）
SLIP_SPREAD_MIN = 1.0    # 价差下限（bp）
SLIP_SPREAD_MAX = 30.0   # 价差上限（bp）
SLIP_IMPACT_K = 150.0    # 平方根冲击系数（bp）
SLIP_CAP = 0.05          # 参与率封顶 5%，超过视为流动性不足

# 股息红利税（差别化）：持股 ≤1月 20%、1月~1年 10%、>1年 0%。
DIV_TAX = True

# 红利再投资（仅 ADJ=False 时生效）
#   True  = 除权日当天按不复权收盘价立即买入同一只股票（CRSP 总收益口径）
#   False = 分红进现金，等下次调仓再配置
#
# 区别在于「现金闲置期」：月调仓下最长闲置 20 个交易日。
# 实测不复权+现金 11.54% vs 前复权 11.08%，
# 前复权是自动立即再投资的，所以要做同口径对比必须开 True。
DIV_REINVEST = True

STAMP_TAX = 0.001
MIN_LIST_DAYS = 250
BETA_WIN = 120
# 市场代理：只用于算 Beta 和 MACD 风格切换，保持沪深300（标准做法）
BENCH = '000300.SH'
# 业绩对比基准：红利策略该和红利指数比，而不是和宽基比
#   上证红利 000015 —— 2014-10-13~2026-10-09 涨 83.49%，与用户给的截图基准 82.37% 吻合
#   可选：000922 中证红利（+112.84%）、H30269 红利低波（+114.70%）
# 与 BENCH 分开，换对比基准不会改动选股逻辑（实测换 Beta 基准只影响 ±0.2pp）
BENCH_CMP = '000015.SH'
# 价格口径：True=前复权（收益率含分红再投资），False=不复权真实价+分红入账
#
# 关于「前复权有没有未来函数」，要分成三件事看（C-12 / C-28）：
#   · 算收益率：P_adj(t2)/P_adj(t1)-1 就是含分红再投资的真实收益，**无未来函数**
#   · 成交/持仓估值：同一时点比较，**无未来函数**
# 两种口径实测：前复权 892.02%（年化 19.56%）vs 不复权+现金分红 876.00%（19.41%），
# 只差 0.15pp/年 —— 就是「分红再投资 vs 现金不复投」的复利，不是 bug。
# 默认口径 = 不复权 + 红利再投资 + 扣红利税（最接近实盘）
# 要看理论税前总收益时才加 --adj
ADJ = '--adj' in sys.argv

# 口径标签统一从这里取，避免各处的字符串不一致
MODE_LABEL = '前复权（税前）' if ADJ else '不复权+红利再投资（税后）'

# 本次回测的「标签」——前端和 runs/ 列表都要显示它，
# 否则跑了一堆实验之后完全分不清哪个是正式结果、哪个是参数扫描。
#
# 优先级：--tag 参数 > 环境变量 QUANT_TAG > 自动推断
#   自动推断：
#     · backtest_fast.py / run.py  → 「正式回测」
#     · 被其它脚本 import 调用     → 取调用者脚本名（sweep_xxx / compare_xxx…）
def _detect_tag():
    for i, a in enumerate(sys.argv[:-1]):
        if a == '--tag':
            v = sys.argv[i + 1].strip()
            if v:
                return v
    v = os.environ.get('QUANT_TAG', '').strip()
    if v:
        return v
    base = os.path.basename(sys.argv[0]).replace('.py', '')
    if base in ('backtest_fast', 'run', '__main__', ''):
        return '正式回测'
    return base or '未命名'


RUN_TAG = _detect_tag()
#   adj = 前复权价（与聚宽 196.69% 的结果吻合）
#   raw = 不复权真实价（口径更"干净"，但会偏离聚宽）
SAVE = True          # 扫描时关掉，不覆盖正式结果文件
# 撮合时点：1 = 用前一根 bar 的数据选股、当日开盘成交（聚宽口径，无未来函数）
#           0 = 用当日收盘数据选股、当日开盘成交（旧版，含未来数据）
EXEC_LAG = 1
# Beta 合理区间。
# 曾经设成 0.05~1.60，理由是「负 Beta 是停牌/跳价造成的协方差失真」——实测证伪：
#   · 不过滤           191.49%，回撤 -23.19%，Calmar 1.03
#   · 0.05~1.60（错）  121.13%，回撤 -23.05%，Calmar 0.75   ← 少赚 70pp，回撤没改善
#   · -0.5~2.0（现在）  212.16%，回撤 -23.04%，Calmar 1.11
# 原因：轻微负 Beta 多是公用事业/水电（三峡能源 -0.39），下跌市里天然抗跌，
# 恰恰是真红利；只有 -6 那种极端值（冀中能源）才是真噪音。

# Beta 算法：
#   'jq'     复刻聚宽 jqfactor（252 日 / 全A市值加权 / 指数加权 halflife=63 / 停牌收益=0）
#   'simple' 本地旧算法（120 日 / 沪深300 / 等权 rolling / 停牌收益=NaN）
# 对齐聚宽验证时必须用 'jq'；只看「理论更正确」可用 'simple'。
BETA_MODE = 'jq'

# Beta 的市场加权用「流通市值」还是「总市值」。
#   理论上聚宽是流通市值，但实测改用流通市值后本地结果反而更差
#   （2016-2026 全约束：144.68% → 125.13%），且低波线选股一只没变。
#   原因：本地流通比例是「当前快照」反推的，对历史是近似；
#        且银行占流通市值权重本来就高，改成流通市值并没有让工行 Beta
#        低到能进前 4（总市值 1.015 / 等权 0.611，而瀚叶 0.378）。
#   默认关闭，保留开关供后续验证。
BETA_USE_FLOATCAP = False

#   True  = 每股法（正确口径）  dy = 每股派息 / 股价
#   False = 复刻聚宽（有缺陷）  dy = 每股派息 / 股价 × 流通比例
#
# 聚宽的算法是：
#            = (每股派息 × 流通股本) / (股价 × 总股本)
#
# 实测（聚宽环境，2015-12-31）：
#     北新建材  聚宽记 3.00 亿 / 实际 6.01 亿 = 0.50（流通比例）
#     瀚叶股份  聚宽记 1.62 亿 / 实际 4.38 亿 = 0.37（流通比例）
#     工商银行  聚宽记 910 亿  / 实际 910 亿  = 1.00（全流通）
#     国投电力  聚宽记 19.6 亿 / 实际 19.6 亿 = 1.00（全流通）
#
#      （瀚叶真实 3.82% 被算成 1.41%，北新真实 3.57% 被算成 1.79%）
#      这就解释了为什么聚宽漏掉了这两只、而本地选到了它们。
DY_USE_FLOAT_SHARES = True

# select() 用的「流通比例矩阵」(T, N)，由 run() 在构建 CAPM 时顺手填好。
_RATIO_M = None

# 直接用聚宽算好的因子（dividend_yield / beta / pe / 三项增速）。
#
# 为什么需要这个开关：
#     不是一回事。聚宽的全局中位只有 0.56%、>3% 仅占 10%，
#     而本地算出来中位约 1%、>3% 占 20%+ —— 候选池大小差一倍。
#
#     要让「选股逻辑」和聚宽完全一致，只能直接用它的因子值。
#     数据来源：聚宽研究环境导出（jq_export_factors.py → jq_factors.csv）
#              → ingest_jq_factors.py → clean_pipe/jq_factors.parquet
#
# True 时：dy / beta / pe / 三项增速 / cash_per_share 全部用聚宽的值
# False 时：用本地自己算的（默认，实盘可用）
USE_JQ_FACTORS = False

# 细粒度开关：分别控制用不用聚宽的某一个因子。
#   USE_JQ_FACTORS=True 时，下列为 None 表示「全部因子都用聚宽的」；
#   设为集合（如 {'dy'}）则【只有】集合里的因子用聚宽值，其余用本地算的。
#
#   用途：把「本地 vs 聚宽」的 11pp 差距拆到单个因子上，
#        看是 dy / beta / pe 哪个贡献最大。
#        之前一刀切 USE_JQ_FACTORS 只能回答「全换差多少」。
JQ_FACTOR_SET = None

# 用聚宽导出的「基础池」替代本地 pool。
#
# 本地 pool = alive(有价) & listed(上市满250天) & tradable(有成交量)
# 聚宽 base_pool = 主板创业板 & 上市满250天 & 非ST & 非停牌
#
# 实测 2015-12-31：聚宽 2336 只，本地 2135 只，差 201 只。
# 这个差异会改变「前 10%」的阈值，进而改变整个低波候选池。
#
# True 时：直接用 jq_factors.parquet 里那期的 code 集合当 pool，
#          排除「基础池不同」这个变量，验证「选股逻辑本身」是否一致。
USE_JQ_POOL = False

# 聚宽因子表 {date_str: DataFrame(code, dividend_yield, beta, ...)}，run() 里加载
_JQ_F = None

# 成长增速口径：
#   True  = 单季度同比（聚宽 indicator 口径，用 income_yoy.parquet）
#   False = 累计同比（本地 fin 的 FN183/FN184/FN189）
#
# 两者差别很大（华电国际 2015Q3：累计 -9.97%，单季 -26.40%），
# 价值线的「营收 5~100%」会筛出完全不同的股票。
# 对齐聚宽时必须用 True。
USE_SQ_YOY = True

# ---- 红利质量硬门槛（对齐中证红利低波 / 浙商研报 / 景顺"假红利四分类"）----
# 股利增长率门槛：最近一个完整年度派现 ÷ 上一个完整年度派现 - 1
#   None = 关闭；0 = 要求不下降；0.1 = 要求增长 10% 以上
# 中证红利低波把「股利增长率非正」直接剔除。C-20 的间接证据（TTM 赢三年平均）
# 暗示「分红在提升」可能是真信号，这条单独验证。
DIV_GROWTH_MIN = None
#   分子 FN107 是「最近一期累计经营现金流」（可能 3/6/12 个月，取决于该公司公告到哪一期），
#   分母是「过去 365 天派现」（12 个月）—— 时间跨度不一致，覆盖倍数在 0.09~10.6 之间乱跳。
#   实测阈值 1.0 能跑出 434%，但它实际筛的是「公司是否已披露年报」，不是现金流健康度。
# 正确做法：分母换成同一报告期的归母净利润（FN96），口径天然一致。
# 这就是浙商六因子里的 A3「经营现金流/扣非净利润」。
# 阈值取 1.0：经营现金流不低于净利润。选 1.0 不是因为跑出来最高（0.8 更高），
# 而是它本身就有明确含义，且 Calmar 最高（1.80），不是调出来的数字。
OCF_NP = 1.0
# 在 2014-2026 + 12 只下翻案：1189.42%（基线 935.73%），Calmar 0.47→0.52，

# 本地一直没有这个下限，是最明显的一处「没对齐」。
#       —— 那不是红利股，是「同组里相对高」而已。
DY_MIN = 0.0

# 板块白名单：只保留聚宽原策略覆盖的主板 + 创业板。
# 聚宽的 `is_main_or_gem()` 只认 600/601/603/605/000/001/002/003/300/301，
# 明确剔除科创(688/689)、北交(4/8/920)、B股(2/9)。
# 本地默认全市场（5,224 只），科创板会被选进红利组合 —— 那是错的：
# 科创板 2019 年才有、大多不分红、波动远大于主板，不是红利策略的猎场。
BOARD_MAIN_GEM = False   # True = 只用主板+创业板（对齐聚宽）


# ---- 选样空间约束（目标已从"复刻红利"改为"高收益低回撤"，这些是硬筛子）----
# 清单 11.2 华福「红利+」：为防小市值偏移，只在中证800 + 中证1000 成分股里挑。
# 官方成分数据只有 2020 后的季度快照且不完整（每期 97/146/276 只，不是 300/500/1000），
# 改用「当期总市值排名前 N 名」近似：800≈中证800，1800≈中证1800（华福口径）。
POOL_TOP_N = 0
# 行业黑名单。曾以为剔地产没影响（旧区间 118.53% vs 120.79%），
# 新区间翻案：1054.28% vs 935.73%，Calmar 0.47→0.49，两段都改善。
# 「潜伏高股息」增强：在分红含权期（预案公告日 → 除权日）内，候选池里优先选含权股。
# 且 5~7 月每月有 800~2600 只含权股，11~1 月只有 12~40 只，所以只在旺季启用。
# 实测：全年启用最好（+232pp / 回撤改善 1.1pp / Calmar 0.53→0.58），
# 不是季节性——"能进入含权期"本身就说明公司确实在分红，是个持续的正向信号。
# 安慰剂检验：随机打乱同比例矩阵 5 次，均值 1073%（低于基线 1218%），
# 真实 1450% 超额 +377pp，随机组 0/5 次超过——信号有信息量。
EP_BOOST = True
EP_MONTHS = tuple(range(1, 13))   # 全年；想只做旺季可改成 (4,5,6,7)
#   0 = 不限（只要含权就优先，C-24 的做法）
EP_MIN_DY = 0.0
# 距除权日的天数过滤（配合含权优先）：
#   EP_MAX_DAYS = 9999 → 不限（只要在含权期就优先）
#   越小 → 只让「快要除权」的股票优先；EP_MIN_DAYS 用来排除「马上要除权」的
# C-23 只测过「预案→除权」的总长度，没测过「当前距除权多远」。
EP_MAX_DAYS = 9999
EP_MIN_DAYS = 0
_EP_DAYS = None          # (T, N) int16：距下次除权的天数，不在含权期为 9999
_EP = None               # (T, N) 布尔矩阵：该日该股是否处于含权期
EP_OVERRIDE = None       # 安慰剂检验用：非 None 时直接用这个矩阵，不自己构建

IND_MAX = 0              # 单一申万一级行业最多持有几只；0=不限

# 择时信号源：
#   macd     沪深300 月线 MACD 柱（原策略自造，全网找不到出处）
#   bond     股债信号·广发方向：10Y国债期货价格 z-score 高（=利率低=债贵）→ 防守
#   bond_inv 股债信号·中金方向：z 高（利率低）→ 红利占优 → 进攻
#   none     不择时，永远进攻（用于检验择时到底有没有贡献）
TIMING = 'macd'
BOND_WIN = 250       # 债券价格 z-score 的滚动窗口
# 涨停开板检查（原策略每日 11:00 的日内逻辑）
#   False = 不检查（本地默认，A股涨跌停极少触发，见 check_tradability.py）
#   True  = 日线近似：昨日封板 + 今日开盘已开板 + 不在目标池 → 卖出
LIMIT_OPEN_CHECK = False

# 涨跌停成交检查：涨停买不进、跌停卖不出。
#   聚宽的 order_value_safe 有这道检查，本地一直没有 ——
#   回测里「涨停当天买入」「跌停当天卖出」都是假成交。
#   数据源 extra/stopprice.parquet（5,224 只 / 1,681 万行 / 1990 起）。
LIMIT_CHECK = False
# 调仓频率（月）：1=月度（原策略）/ 2=双月 / 3=季度 / 6=半年
# 实测保持月度。降频收益非单调（双月 868%→季度 712%→4个月 1241%→半年 617%），
# 是锯齿不是趋势，属噪音；而"省交易成本"这个理由不成立——
# 每笔只交易 1/12 仓位，印花税+佣金的年化拖累只有约 0.05%，可忽略。
REBAL_FREQ = 1
#   ttm = 过去 365 天派现 ÷ 当前真实价（我们一直用的）
#         好处：单次特别分红或某一年的股价异动不会把排名顶飞
DY_MODE = 'ttm'
_YEAR_END_PR = {}        # year -> 该年最后一个交易日的真实收盘价数组
MAX_RUNS = 100           # 前端「历史回测」最多保留多少次
# 低波线占总持仓的比例（进攻时按此比例，防守时取补数）
LOW_RATIO = 1.0 / 3.0

# ---- 权重体系 ----
#   equal = 等权（一直用的做法，12 只各 8.3%）
# 加权后权重会分化，MAX_WEIGHT（清单 2.4/6.6 单只上限）才有意义
WEIGHT_MODE = 'equal'
MAX_WEIGHT = 0.0         # 单只权重上限，0=不限；例如 0.15
_WEIGHTS = {}            # code -> 目标权重（由 select 写入，rebalance 读取）

# ---- 目标波动率缩放（Moreira & Muir 2017）----
#   exposure = min(1, VOL_TARGET / 过去 VOL_WIN 日净值年化波动)
# 不预测方向，只在波动放大时自动收缩敞口。
#
# 出处（都是 A 股实证）：
#   中泰「动量亦有时」126日RV目标25%：回撤 -56.6%→-41.9%，夏普 0.48→0.63
#   GitHub 行业ETF动量：敞口=min(1,15%/126日RV)，月频调
#   GitHub A股因果盲测：20日RV → +20.61% / -12.70% / 摩擦仅1%
# 同一份清单明确「不要做」固定%/ATR止损 —— 17 个标的里 10 个比躺平回撤更深。
#
# 净值后处理实测（126日、月频）：Calmar 0.50 → 0.90，回撤 -41.68% → -23.09%，
# 且目标 10~25%、窗口 20~250 日两个维度都是连续平台（不是锯齿）。
VOL_TARGET = 0.0         # 目标年化波动率；0=关闭，0.18=18%
VOL_WIN = 126            # 滚动窗口（交易日）
VOL_EXPO_MIN = 0.30      # 最低仓位，避免极端行情下完全空仓
_VOL_EXPO = 1.0          # 当前敞口（run 里算好，rebalance 读）
# 每日检查（而不是只在月度调仓日检查）。
# 只挂在调仓日的话，2015-06 波动在 6 月中下旬飙升，要等到 7 月初才降仓，
# 正好错过最惨的一周 —— 这是上一版失效的直接原因。
VOL_DAILY = True         # True=每天检查敞口偏差，False=只在调仓日
VOL_BAND = 0.15          # 敞口偏差超过多少才动手（0.15=15 个百分点）
_VOL_N_ADJ = 0           # 统计日频调整触发了几次

# ---- 回撤止损（另一种控制暴跌的方式，和 VOL_TARGET 二选一或叠加）----
# 从历史高点回撤超过 DD_STOP → 把仓位降到 DD_EXPO；
# 从低点反弹超过 DD_RESUME → 恢复满仓。
# 之前用净值序列变换测过「25% 止损」Calmar 0.83，但那是零成本日频假象；
# 这里进引擎跑，含真实成本与调仓摩擦。
DD_STOP = 0.0            # 0=关闭；0.25=回撤 25% 触发
DD_EXPO = 0.0            # 触发后的目标仓位（0=清仓，0.5=减半）
DD_RESUME = 0.10         # 从低点反弹该幅度后回场
_DD_TRIG = False         # 当前是否处于止损状态
_DD_LOW = 1.0            # 止损期间的最低净值
_DD_N = 0                # 触发次数

# ---- 双均线仓位择时 ----
# 信号：基准（上证红利）收盘 > MA_fast 且 > MA_slow → 满仓；否则降到 MA_EXPO。
# 用 T+1 执行（今天收盘算信号，明天开盘调仓），与 EXEC_LAG 一致。
#
# 净值后处理实测（零成本、日频）：
#   满仓       1019.17% / 20.69% / -41.68% / Calmar 0.50
#   close>20 且 close>60  456.48% / 14.30% / -18.61% / Calmar 0.77，切换 297 次
# 但零成本日频是虚高（C-35 教训：0.90 进引擎变 0.61），所以要在引擎里重跑。
# ---- LPPL 泡沫信号（只防泡沫型崩盘）----
# 沪深300 的 LPPL 拟合（250 日窗口、5 日一次），临界点距离 tc<15 日才触发。
# 净值变换实测（零成本）：1272.47% / 22.63% / -31.55% / Calmar 0.72
# 触发只有 2 段：2015-06-24~07-28、2018-01-18~03-07，其余 11 年完全不动。
# 信号已 shift(1)，存在 lppl_signal.parquet。
LPPL_TIMING = True       # True=启用 LPPL 泡沫减仓
LPPL_EXPO = 0.30         # 泡沫期仓位（不能设 0：清仓后 holding 空，无法加回）
LPPL_BAND = 0.10
_LPPL_ARR = None

MA_TIMING = False        # True=启用双均线仓位择时
MA_FAST = 20             # 快线
MA_SLOW = 60             # 慢线
# 信号不满足时的目标仓位。**不能设 0**：清仓后 holding 变空，
# adjust_exposure 无法按比例把仓位加回来（它不知道该买什么），
# 只能干等到下个月调仓日，信号恢复的响应会延迟 20 天。
# 留 5%（约半只票）既接近空仓，又保证 holding 非空、能立刻加回。
MA_EXPO = 0.05
MA_BAND = 0.05           # 仓位偏差超过该值才调（越小越紧跟信号，成本越高）
_MA_ARR = None           # 预计算的信号数组（已 shift(1)）
_MA_N = 0                # 调仓次数
#         也是中证/上证指数公司的标准算法
#   adj = 前复权价。⚠️ 含未来信息：前复权因子以「今天」为基准，扣掉了 t 之后发生的
#         前复权价被压到 7.19 → 算出 25.2%，放大 2.7 倍。
for _a in sys.argv:
    if _a.startswith('--exec='):
        EXEC_LAG = 0 if _a.split('=', 1)[1] == 'same' else 1
    elif _a.startswith('--timing='):
        TIMING = _a.split('=', 1)[1]

CACHE = os.path.join(BASE, 'cache_bars_adj.pkl' if ADJ else 'cache_bars_raw.pkl')


_MEM = {}          # 进程内缓存：参数扫描时只读一次盘


# ---------------------------------------------------------------- 数据
def load_amount():
    """成交额矩阵（**元**）。

    必须从 daily_raw 读：daily/（前复权）只有 7 列、没有 amount。

    ⚠️ 单位曾经搞错过，导致滑点被低估 99%：
       旧注释写「原始单位是万元」，于是调用处 `* 1e4` 转成元。
       但实测（2024-01-02 平安银行）：
           amount = 1,075,742,252   ← 10.76 亿元，本来就是「元」
           volume =     1,158,366   ← 手
           close  =          9.21   ← 元/股
           amount / volume / close = 99.97 ≈ 100（100 股/手）
       再乘 1e4 后，参与率 trade/amt 缩小 10000 倍，
       冲击 150×sqrt(part) 只剩真实的 1/100：
           下单 500 万 → 应为 10.23bp，实际算成 0.10bp。

    现在直接返回元，调用处不再乘 1e4。
    """
    if 'amt' in _MEM:
        return _MEM['amt']
    d = os.path.join(CLEAN, 'daily_raw')
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.parquet'))
    df = pd.concat([pd.read_parquet(f, columns=['date', 'code', 'amount'])
                    for f in files], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(['date', 'code'])
    amt = df.pivot(index='date', columns='code', values='amount')
    _MEM['amt'] = amt
    return amt


def load_bars():
    # 缓存必须按 sub 分开：否则同一次进程里先跑前复权、再跑不复权，
    # 第二次会直接命中第一次的缓存，两种口径实际用的是同一份价格
    sub = 'daily' if ADJ else 'daily_raw'
    _key = 'bars_' + sub
    if _key in _MEM:
        return _MEM[_key]
    stamp = max(os.path.getmtime(os.path.join(CLEAN, sub, f))
                for f in os.listdir(os.path.join(CLEAN, sub)))
    if os.path.exists(CACHE):
        try:
            obj = pd.read_pickle(CACHE)
            if obj.get('sub') == sub and obj.get('stamp') == stamp:
                _MEM[_key] = obj
                return obj
        except Exception:
            pass

    d = os.path.join(CLEAN, sub)
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
    _MEM[_key] = obj
    return obj


RAW_CACHE = os.path.join(BASE, 'cache_raw_close.pkl')


def load_raw_close():
    """不复权收盘价（只取估值用），带缓存。"""
    d = os.path.join(CLEAN, 'daily_raw')
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


def _load_clean_or_raw(name):
    """优先读 data_pipeline 的清洗产物，没有才回退到裸下载目录。

    为什么必须优先清洗版：
        裸数据里有重复主键、量纲错误、未来数据（财务修正），
        不挡住的话「同一个策略换个时间跑结果不一样」，回测不可复现。
    回退是为了兼容：还没跑过管线的环境仍能跑起来（但会打警告）。
    """
    cp = os.path.join(BASE, 'clean_pipe', '%s.parquet' % name)
    if os.path.exists(cp):
        return pd.read_parquet(cp), True
    d = os.path.join(BASE, name)
    files = sorted(os.path.join(d, f) for f in os.listdir(d)
                   if f.endswith('.parquet'))
    print('⚠️ %s 还没走清洗管线，用的是裸数据（python data_pipeline.py %s）'
          % (name, name))
    return pd.concat([pd.read_parquet(f) for f in files],
                     ignore_index=True), False


def load_fin():
    df, cleaned = _load_clean_or_raw('fin')
    df['announce_time'] = pd.to_datetime(df['announce_time'].astype(str),
                                         errors='coerce')
    df = df.dropna(subset=['announce_time'])
    return df.sort_values(['code', 'announce_time'])


def load_div():
    """除权除息明细。

    ⚠️ 必须同时取三类公司行动，不能只要现金分红：
        ① Bonus     每 10 股派息（元）      → 现金入账
        ② ShareBonus 每 10 股送转（股）      → 持股数 ×(1 + sb/10)
        ③ Allotment/AllotPrice 每 10 股配股数/配股价 → 需要额外现金

    只处理 ① 会出大错：10 送 10 时不复权价腰斩 -50%，
    如果持股数不跟着翻倍，组合净值凭空蒸发一半。
    实测 2016 年后有 4,938 次送转（影响 2,795 只股票），
    其中 1,010 次是 10 送 6 以上（价格腰斩级别）。

    官方除权参考价公式：
        除权价 = [(前收 - 现金红利) + 配股价 × 股份变动比例] / (1 + 股份变动比例)
    """
    df, _cleaned = _load_clean_or_raw('div')
    # 清洗产物的日期列叫 ex_date，裸数据叫 Date —— 统一成 Date
    if 'ex_date' in df.columns and 'Date' not in df.columns:
        df = df.rename(columns={'ex_date': 'Date'})

    def num(col, default=0.0):
        if col not in df.columns:
            return np.full(len(df), default)
        v = pd.to_numeric(df[col], errors='coerce').values
        return np.where(np.isfinite(v), v, default)

    out = pd.DataFrame({
        'code': df['code'].values,
        'ex_date': pd.to_datetime(df['Date']).values,
        # 原始数据是「每 10 股」口径，统一除以 10 变成「每股」
        'dps': num('Bonus') / 10.0,
        'share_ratio': num('ShareBonus') / 10.0,     # 每股送转几股
        'allot_ratio': num('Allotment') / 10.0,      # 每股配几股
        'allot_price': num('AllotPrice'),            # 配股价（已是元/股）
    }).dropna(subset=['ex_date'])

    #
    #   实测反例（reverse_dy_missing.py，2016-01-29）：
    #       000425 徐工机械
    #         2015-06-17  Bonus 0.6  ShareBonus 0    ← 只派息
    #         2015-09-23  Bonus 0.0  ShareBonus 20   ← 只送转（10转20）
    #       旧算法（逐条摊薄）：6-17 那天 sb=0 → 不摊薄 → dps_adj = 0.06
    #       但分母 price_f 是「送转后」的价格（小 3 倍）→ dy 虚高 3 倍
    #       旧：dy = 2.0270%   聚宽 0.6757%   比值正好 1/3
    #
    #       新算法（累积摊薄）：6-17 此后有 9-23 的 sb=2.0
    #       dps_adj = 0.06 / 3 = 0.02 → dy = 0.68%  ✅ 和聚宽一致
    #
    #   同批受影响的还有 000426（1/2）、000517（1/3）等 137 只「分两次红」的股票。
    #
    #        分红现金入账（1221 行附近）仍用原始 dps ——
    #        持 100 股、每 10 股派 6 元，实收 60 元，与之后的送转无关。
    out = out.sort_values(['code', 'ex_date']).reset_index(drop=True)
    # ⚠️ 用 log 累加再 exp，避免 cumprod 溢出（历史上有连续 30+ 次送转的股票，
    #    (1+sb) 连乘会超出 float64 范围，报 "overflow encountered in accumulate"）。
    #    同时把累积倍数上限设在 100 倍，超过就认为是脏数据。
    _mult = (1.0 + out['share_ratio']).values
    _logm = np.log(np.clip(_mult, 1.0, 100.0))
    _cum = np.ones(len(out), dtype='float64')
    if len(out):
        _codes = out['code'].values
        _acc = 0.0
        for i in range(len(out) - 1, -1, -1):
            if i == len(out) - 1 or _codes[i] != _codes[i + 1]:
                _acc = _logm[i]
            else:
                _acc = _logm[i] + _acc
            _cum[i] = min(np.exp(min(_acc, np.log(100.0))), 100.0)
    out['dps_adj'] = out['dps'].values / _cum
    return out


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
    """分红索引：按 (code, 公告日) 查询区间内累计派息。"""
    codes = div['code'].tolist()
    _v = div['dps_adj'].values if 'dps_adj' in div.columns else div['dps'].values
    return KeyIndex(codes, div['ex_date'], _v, code_pos)


# ---------------------------------------------------------------- 因子
def month_first_days(dates, freq=1):
    """调仓日：每个自然月首个交易日，再按 freq 抽稀。

    freq=1 月度（原策略）/ 2 双月 / 3 季度 / 6 半年。
    从回测起点的那一个月开始算，避免"能被 freq 整除的月份"这种偏移。
    """
    s = pd.Series(dates)
    g = s.groupby([s.dt.year, s.dt.month])
    firsts = list(g.min())
    if freq <= 1:
        return set(firsts)
    return set(firsts[::freq])


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


def beta_panel_jq(px, raw, cap, vol):
    """复刻聚宽 jqfactor 的 beta 因子（官方文档口径）。

    聚宽官方定义（JQData 因子文档）：
        raw_beta：CAPM 模型中的 β，
                  过去【252 个交易日】股票的收益与市场收益
                  （【全 A 股票收益按流通市值加权】）
                  进行【时间序列指数加权回归】后的斜率系数，
                  【指数加权的半衰期为 63 个交易日】。
                  停牌股票收益率为 0，股票上市需超过 21 个交易日，否则 beta 为 nan。

    和本地旧实现的四个差异（全都改了）：
        ① 窗口      120 → 252（用 ewm 实现，半衰期 63 天自然衰减，无需硬窗口）
        ② 市场基准  沪深300 → 全 A 市值加权
        ③ 加权方式  等权 rolling → 指数加权（halflife=63）
        ④ 停牌      我上一版设成 NaN → 按官方文档设为 0
    """
    ret = px.pct_change()
    # ④ 停牌日收益率 = 0（官方口径，不是 NaN）
    if vol is not None:
        _v = vol.reindex(index=ret.index, columns=ret.columns)
        ret = ret.where((_v > 0) | _v.isna(), 0.0)
    # ⚠️ 不能 fillna(0)！
    #   停牌日是「已上市、当天没成交」→ 收益 0（官方口径）
    #   但「上市前」是「根本不存在」→ 必须保持 NaN，否则
    #   新股上市前的整段历史都被当成 0 收益，EWM 会把 Beta 拉到极低。
    #   （实测 000786 北新建材 2015 年停牌 121 天，fillna 后 Beta 掉到 0.11）

    # ② 市场收益：全 A 按市值加权
    #    聚宽用「流通市值」，本地只有总股本（FN238），用总市值近似。
    #    A+H 公司（银行）会被高估权重，但方向上不影响排序。
    if raw is not None and cap is not None:
        mcap = raw.reindex(index=ret.index, columns=ret.columns) * cap
    else:
        mcap = pd.DataFrame(1.0, index=ret.index, columns=ret.columns)
    w = mcap.where(_v > 0) if vol is not None else mcap
    wsum = w.sum(axis=1)
    ret_m = (ret * w).sum(axis=1) / wsum.replace(0, np.nan)
    ret_m = ret_m.fillna(0.0)

    # ③ 指数加权，半衰期 63 天；min_periods=21 对应「上市需超过 21 个交易日」
    hl, mp = 63, 21
    rm = ret.ewm(halflife=hl, min_periods=mp).mean()
    bm = ret_m.ewm(halflife=hl, min_periods=mp).mean()
    cov = (ret.mul(ret_m, axis=0).ewm(halflife=hl, min_periods=mp).mean()
           - rm.mul(bm, axis=0))
    var = ret_m.ewm(halflife=hl, min_periods=mp).var()
    return cov.div(var, axis=0)


def beta_panel(px, vol=None):
    """一次算出全历史 Beta 面板（对沪深300，BETA_WIN 日窗口）。按窗口缓存，扫描时复用。

    ⚠️ `vol` 必须传！否则停牌日的「假 0 收益率」会把 Beta 压到 0.1 以下。

       清洗时停牌日的 close 被 ffill 成前收（为了不让持仓股"消失"），
       于是 pct_change 在停牌日得到 0。
       这些 0 是假的（那天根本没交易），但它们会真实地进入协方差计算：

           停牌 121 天 / 170 天的 000786，Beta 算出来 0.111
           从不停牌的 601398，Beta 算出来 0.471

       后果：**停牌越多的股票 Beta 越低**，于是永远霸占「低波线」前几名。
       实测 2015-12-31 低波线前 4 名里有 3 个是长期停牌股
       （北新建材 121 天、东方铁塔 144 天、伟星股份 84 天），
       而聚宽选的是工行/华电/招商这类正常交易的票。

       修复：把 volume==0 的日子收益率设为 NaN，让 rolling 自然跳过。
    """
    if BETA_WIN in _BETA:
        return _BETA[BETA_WIN]
    ret = px.pct_change()
    if vol is not None:
        # 停牌 = 当天没成交 → 收益率无意义
        _v = vol.reindex(index=ret.index, columns=ret.columns)
        ret = ret.where((_v > 0) | _v.isna())
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


_STOP_CACHE = {}

# 最近一次 run() 的调仓目标池 [(date, 'code1,code2,...', style), ...]
# 对齐验证 / 持仓分析要用，run() 的返回值里没有它
LAST_PICKS = []

# 选股注入：绕过本地选股，直接用外部给定的目标池。
#
# 用途：用聚宽日志记录的 131 期目标池喂给本地框架，验证
# 「如果选股一致，本地引擎能不能复现聚宽的收益」。
# 这样能把「选股差异」和「交易/成本/估值差异」彻底分开。
#
# 格式：{'YYYY-MM-DD': ['601398.SH', ...], ...}
PICK_OVERRIDE = None


def load_stopprice(dates, codes):
    """涨跌停价矩阵，返回 (涨停价 (T,N), 跌停价 (T,N))。

    ⚠️ 涨跌停价是【不复权】的（交易所按真实价格定），
       所以只能和不复权价比较，不能和 ADJ=True 的前复权价比。

    数据来源：extra/stopprice.parquet（5,224 只 / 1,681 万行 / 1990 起）

    用途：判断「信号日能否成交」——
        涨停买不进、跌停卖不出，这是回测里最大的假成交来源之一。
        聚宽的 order_value_safe 有这道检查，本地一直没有。
    """
    key = (str(dates[0]), str(dates[-1]), len(codes), hash(tuple(codes[:50])))
    if key in _STOP_CACHE:
        return _STOP_CACHE[key]

    p = os.path.join(BASE, 'extra', 'stopprice.parquet')
    if not os.path.exists(p):
        return None, None
    d = pd.read_parquet(p)
    d['date'] = pd.to_datetime(d['time']).dt.normalize()
    d = d[(d['date'] >= dates[0]) & (d['date'] <= dates[-1])]
    up = (d.pivot_table(index='date', columns='code', values='涨停价',
                        aggfunc='last')
          .reindex(index=dates, columns=codes).values.astype('float64'))
    dn = (d.pivot_table(index='date', columns='code', values='跌停价',
                        aggfunc='last')
          .reindex(index=dates, columns=codes).values.astype('float64'))
    _STOP_CACHE[key] = (up, dn)
    return up, dn


def limit_up_matrix(PR, dates, codes):
    """(T, N) 每个交易日每只股票的涨停价。

    必须用**不复权价**算——涨跌停是按真实价格定的；
    成交仍按前复权价（等比例，收益率一致）。
    分板块规则：主板 10%、创业板 10%→2020-08-24 起 20%、科创板 20%、北交所 30%。
    """
    T, N = PR.shape
    prev = np.vstack([np.full((1, N), np.nan), PR[:-1]])
    cyb = np.array([c.startswith(('300', '301')) for c in codes])
    kcb = np.array([c.startswith(('688', '689')) for c in codes])
    bj = np.array([c[0] in ('4', '8') for c in codes])
    rate = np.full(N, 0.10)
    rate[kcb] = 0.20
    rate[bj] = 0.30
    R = np.tile(rate, (T, 1))
    cut = np.asarray(dates >= pd.Timestamp('2020-08-24'))[:, None]
    R = np.where(cut & cyb[None, :], 0.20, R)
    return np.round(prev * (1 + R), 2)


def check_limit_open(row, d, holding, cash, O, up_mat, sealed,
                     code_pos, target_pool, trades):
    """涨停开板检查（日线近似：把当日开盘价当成 11:00 的价格）。

    原策略：昨日封板的持仓股，次日 11:00 若已开板且不在目标池 → 卖出 → 补位。
    这里只做卖出，不补位（补位要另外的钱，先看纯效果）。
    """
    if row <= 0:
        return holding, cash
    for c in list(holding.keys()):
        if holding.get(c, 0) <= 0 or c in target_pool:
            continue
        j = code_pos.get(c)
        if j is None:
            continue
        if not sealed[row - 1, j]:          # 昨日没封板
            continue
        op, up = O[row, j], up_mat[row, j]
        if not (np.isfinite(op) and np.isfinite(up)):
            continue
        if op >= up - 0.011:                # 一字板，继续持有
            continue
        sh = holding.pop(c)
        amt = sh * op
        cash += amt - (max(5.0, amt * (COMMISSION + TRANSFER_FEE)) + amt * STAMP_TAX)
        trades.append((d, c, 'sell', sh, op))
    return holding, cash


def bond_signal(dates):
    """股债信号：10 年期国债期货 T0 收盘价的滚动 z-score。

    z > 0 表示债券价格处在近期高位，即利率处在低位（债贵股相对便宜）。
    方向交给调用方决定：
      TIMING='bond'     → 取负（z 高视为防守）
      TIMING='bond_inv' → 取正（z 高视为进攻）
    返回的序列与 select() 里 hist 的用法一致：> 0 判为 red（进攻）。
    """
    p = os.path.join(BASE, 'bond.parquet')
    if not os.path.exists(p):
        print('警告：缺 bond.parquet，择时降级为「不择时」。先跑 python fetch_bond.py')
        return None
    b = pd.read_parquet(p)
    t = b[b['symbol'] == 'T0'].set_index('date')['close'].sort_index()
    if not len(t):
        return None
    mp = max(20, BOND_WIN // 2)
    ma = t.rolling(BOND_WIN, min_periods=mp).mean()
    sd = t.rolling(BOND_WIN, min_periods=mp).std()
    z = (t - ma) / sd
    z = z.replace([np.inf, -np.inf], np.nan).reindex(dates).ffill()
    coverage = z.notna().mean()
    print('股债信号：T0 %d 根 ｜ 覆盖回测期 %.0f%% ｜ 末值 z=%.2f'
          % (len(t), coverage * 100, z.dropna().iloc[-1] if z.notna().any() else float('nan')))
    return z


def macd_monthly(bench):
    m = bench.resample('ME').last().dropna()
    dif = m.ewm(span=12, adjust=False).mean() - m.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    return (dif - dea) * 2


# ---------------------------------------------------------------- 主流程
def run():
    progress.start(1, params={'实验': RUN_TAG, '口径': MODE_LABEL,
                              '目标只数': TARGET_NUM, '缓冲': BUFFER,
                              'Beta窗口': BETA_WIN, '区间': '%s~%s' % (BT_START, BT_END)})
    t0 = time.time()
    progress.phase('加载行情数据')
    bars = load_bars()
    px, op, vol = bars['px'], bars['op'], bars['vol']
    # 成交额（元）：单独从 daily_raw 读。
    # ⚠️ 这里曾经 `* 1e4`（以为原始单位是万元），实测原始就是元，
    #    多乘 1e4 让参与率缩小 10000 倍、冲击成本只剩 1/100。
    #    实测：下单 500 万时，应收 10.23bp 却算成 0.10bp。
    AM = None
    if SLIP_ON:
        try:
            _amt = load_amount()
            AM = (_amt.reindex(index=px.index, columns=px.columns)
                  .values.astype('float64'))
            print('成交额矩阵：有效 %.0f%% ｜ 中位 %.2f 亿元'
                  % (100 * np.isfinite(AM).mean(), np.nanmedian(AM) / 1e8))
        except Exception as e:
            print('警告：成交额加载失败（%s），滑点降级为固定 32bp' % str(e)[:60])
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

    # 涨跌停判断用的「不复权开盘价」。
    #   ADJ=False 时 O 本身就是不复权价，直接复用；
    #   ADJ=True 时 O 是前复权价，和涨跌停价（不复权）不在同一坐标轴，
    #   必须单独读一份真实开盘价。
    if ADJ and LIMIT_CHECK:
        _raw_open = []
        _rp = os.path.join(CLEAN, 'daily_raw')
        for _f in sorted(os.listdir(_rp)):
            if not _f.endswith('.parquet'):
                continue
            _d = pd.read_parquet(os.path.join(_rp, _f),
                                 columns=['date', 'code', 'open'])
            _d['date'] = pd.to_datetime(_d['date'])
            _raw_open.append(_d)
        _ro = (pd.concat(_raw_open, ignore_index=True)
               .pivot(index='date', columns='code', values='open')
               .reindex(index=dates, columns=codes))
        O_RAW = _ro.values.astype('float64')
        del _raw_open, _ro
    else:
        O_RAW = O

    UP_M, DN_M = (None, None)
    if LIMIT_CHECK:
        UP_M, DN_M = load_stopprice(dates, codes)
        if UP_M is not None:
            _cov = np.isfinite(UP_M).mean() * 100
            print('涨跌停检查：开启（覆盖 %.0f%%）' % _cov)

    # 含权期矩阵（预案公告日 <= t <= 除权日）+ 距除权天数矩阵
    global _EP, _EP_DAYS
    _EP = None
    _EP_DAYS = None
    if EP_BOOST and EP_OVERRIDE is not None:
        # 安慰剂检验用：外部塞进来的（可能是随机打乱的）矩阵，不再自己构建
        _EP = EP_OVERRIDE
        print('含权期矩阵：使用外部注入 %s' % str(_EP.shape))
    elif EP_BOOST:
        bp_path = os.path.join(BASE, 'bonus_plan.parquet')
        if os.path.exists(bp_path):
            bp = pd.read_parquet(bp_path).dropna(subset=['plan_date', 'ex_date'])
            bp = bp[(bp['dps'] > 0) & (bp['ex_date'] >= '2013-01-01')]
            ep = np.zeros((len(dates), len(codes)), dtype=bool)
            epd = np.full((len(dates), len(codes)), 9999, dtype=np.int16)
            dvals = dates.values
            hit = 0
            for _, r in bp.iterrows():
                c = r['code'] + ('.SH' if str(r['code'])[0] == '6' else '.SZ')
                j = code_pos.get(c)
                if j is None:
                    continue
                i0 = np.searchsorted(dvals, np.datetime64(r['plan_date']))
                i1 = np.searchsorted(dvals, np.datetime64(r['ex_date']))
                i1 = min(i1, len(dates) - 1)
                if i1 >= i0:
                    ep[i0:i1 + 1, j] = True
                    # 同一股票同一时点可能有多条记录（年报+中期），取最近的除权日
                    left = (r['ex_date'] - dates[i0:i1 + 1]).days.values.astype(np.int16)
                    np.minimum(epd[i0:i1 + 1, j], left, out=epd[i0:i1 + 1, j])
                    hit += 1
            _EP = ep
            _EP_DAYS = epd
            print('含权期矩阵：%d 条记录 ｜ 覆盖 %.1f%% 的股票-日 ｜ 距除权中位 %d 天'
                  % (hit, 100.0 * ep.mean(), int(np.median(epd[ep])) if ep.any() else 0))
        else:
            print('警告：缺 bonus_plan.parquet，含权优先降级为关闭（先跑 fetch_bonus_plan.py）')

    _YEAR_END_PR.clear()
    if DY_MODE == '3y' and PR is not None:
        for y in range(dates[0].year, dates[-1].year + 1):
            pos_y = np.flatnonzero(dates.year == y)
            if len(pos_y):
                _YEAR_END_PR[y] = PR[pos_y[-1]]

    P0 = np.isfinite(P)
    first_idx = np.argmax(P0, axis=0)

    progress.phase('计算 Beta 面板')
    t0 = time.time()
    if BETA_MODE == 'jq':
        # 总股本矩阵（静态快照）：每只股票取最新一期 FN238。
        # 聚宽用「流通市值」，我们只有总股本，用总市值近似。
        # ⚠️ 必须用【全市场 + 流通股本】构建市场收益，不能用总股本、更不能只用手头几只股票。
        #
        #   实测（diag_beta_hist.py，asof=2016-01-04）：
        #       用 8 只样本构造市场收益  →  工行 Beta = 1.015
        #       用全市场 3904 只         →  工行 Beta = 0.335
        #   差 3 倍。而聚宽的低波线前 4 里有工商银行
        #   （说明它的工行 Beta < 0.4），只有「全市场 + 流通股本」才能对上。
        #
        #   数据来源：QMT 的 Capital 表（含 circulating_capital 历史真值）
        #            → data_pipeline.py 清洗 → clean_pipe/capital.parquet
        _cp = os.path.join(BASE, 'clean_pipe', 'capital.parquet')
        CAPM = None
        if os.path.exists(_cp):
            _cap = pd.read_parquet(_cp)
            _cap['ann_date'] = pd.to_datetime(_cap['ann_date'], errors='coerce')
            _cap = _cap.dropna(subset=['ann_date'])
            _col = ('circulating_capital' if BETA_USE_FLOATCAP
                    else 'total_capital')
            _pv = (_cap.pivot_table(index='ann_date', columns='code',
                                    values=_col, aggfunc='last')
                   .reindex(columns=codes))
            _pv = _pv.reindex(_pv.index.union(dates)).ffill().reindex(dates)
            CAPM = _pv.values.astype('float64')
            _cov = np.isfinite(CAPM).mean() * 100
            print('Beta 加权：%s ｜ 覆盖 %.0f%% ｜ %d 只有股本'
                  % ('流通市值' if BETA_USE_FLOATCAP else '总市值', _cov,
                     int(np.isfinite(_cap[_col]).sum() > 0 and
                         _cap['code'].nunique())))

            #   实测北新 0.50、瀚叶 0.37、工行 1.00、国投 1.00，完全对应。
            global _RATIO_M
            _pv_c = (_cap.pivot_table(index='ann_date', columns='code',
                                      values='circulating_capital',
                                      aggfunc='last')
                     .reindex(columns=codes))
            _pv_c = (_pv_c.reindex(_pv_c.index.union(dates))
                     .ffill().reindex(dates))
            _pv_t = (_cap.pivot_table(index='ann_date', columns='code',
                                      values='total_capital',
                                      aggfunc='last')
                     .reindex(columns=codes))
            _pv_t = (_pv_t.reindex(_pv_t.index.union(dates))
                     .ffill().reindex(dates))
            with np.errstate(invalid='ignore', divide='ignore'):
                _RATIO_M = (_pv_c / _pv_t).values.astype('float64')
            _RATIO_M = np.where(np.isfinite(_RATIO_M) & (_RATIO_M > 0)
                                & (_RATIO_M <= 1.0), _RATIO_M, 1.0)
            print('流通比例矩阵：中位 %.3f' % float(np.nanmedian(_RATIO_M)))
        else:
            print('⚠️ 缺 clean_pipe/capital.parquet，回退到 FN238 总股本')
            _cap_s = (fin.assign(_c=pd.to_numeric(fin['FN238'], errors='coerce'))
                      .sort_values('announce_time')
                      .groupby('code')['_c'].last())
            CAPM = np.tile(_cap_s.reindex(codes).values, (len(dates), 1))
        # ⚠️ Beta 的价格序列用【前复权价】，不是不复权价。
        #
        #   试过不复权（保留除权日的 -65% 跳空），结果更差：
        #       不复权  年化 15.91%  回撤 -31.01%
        #       前复权  年化 17.19%  回撤 -27.45%
        #   原因：东方铁塔 2015-12-17「复牌 + 10 送 20」双重叠加，
        #        不复权价从 26.84 直接跳到 9.27（-65%），
        #        这不是真实亏损，是股本变动 + 停牌期间累积的调整。
        #
        #   而且聚宽官方明确写着「为保证数据的连续性，所有数据基于后复权计算」
        #   （JQData 使用说明），后复权连续 = 无跳空 = 和前复权对收益率等价。
        #   所以前复权是对的。
        beta = beta_panel_jq(px, raw if raw is not None else px, CAPM, vol)

        # ⚠️ 加「市值加权标准化 + 2.5σ 去极值」，让 beta 的数值范围对齐聚宽。
        #
        #   逆向验证（jq_reverse_beta.py / 聚宽研究环境实测）：
        #       本地 raw CAPM  beta 范围  0.3 ~ 1.2      中位 0.95
        #       聚宽 jqfactor beta 范围 -2.2 ~ +2.8      中位 0.79  均值 0.64
        #       两者「值相关 0.96+」，但绝对值差 6 倍。
        #
        #   聚宽的变换（7 只样本拟合 + 全市场分位数验证）：
        #       z = (raw - wmean) / std          ← wmean 用流通市值加权
        #       z = clip(z, -2.5, +2.5)          ← 0% 和 1% 分位都是 -2.1966
        #   实测聚宽：等权均值 0.6419 ｜ 市值加权均值 -0.0836（≈0，证明是加权标准化）
        #
        #   ⚠️ 这一步【不改变排序】（单调线性变换 + 端点截断），
        #      所以对低波线选股没有影响，只是让「打印出来的 beta 值」
        #      和聚宽能直接对比，调参时不用再换算。
        if CAPM is not None:
            _w = np.where(np.isfinite(CAPM) & (CAPM > 0), CAPM, 0.0)
            _wsum = _w.sum(axis=1, keepdims=True)
            _w = np.divide(_w, _wsum, out=np.zeros_like(_w), where=_wsum > 0)
            _bv = beta.values.astype('float64')
            _wm = np.nansum(np.where(np.isfinite(_bv), _bv * _w, 0.0), axis=1)
            _sd = np.nanstd(_bv, axis=1)
            _sd = np.where(np.isfinite(_sd) & (_sd > 1e-9), _sd, np.nan)
            _z = (_bv - _wm[:, None]) / _sd[:, None]
            _z = np.clip(_z, -2.5, 2.5)
            beta = pd.DataFrame(_z, index=beta.index, columns=beta.columns)
            print('Beta 标准化：市值加权均值 + 等权std + 2.5σ 截断（对齐聚宽）')
        print('Beta 算法：聚宽口径（252日/全A加权/指数加权63天/前复权价）')
    else:
        beta = beta_panel(px, vol)   # 旧算法：120日/沪深300/等权
        print('Beta 算法：本地简单口径（120日/沪深300/等权）')
    # 挂到模块级，供 diff_beta_full.py 等诊断脚本做全市场对比
    global LAST_BETA
    LAST_BETA = beta
    beta_v = beta.reindex(columns=codes).values if beta is not None else None
    vol_v = vol_panel(px).reindex(columns=codes).values
    print('Beta 面板 %.1fs' % (time.time() - t0))

    progress.phase('建财务/分红索引')
    t0 = time.time()
    fin_idx = build_fin_index(fin, code_pos)
    div_idx = build_div_index(div, code_pos)

    # 聚宽因子表（可选）
    global _JQ_F
    _JQ_F = None
    if USE_JQ_FACTORS:
        _JQ_F = load_jq_factors()

    # 单季度同比索引（聚宽 indicator 口径）
    sq_idx = None
    if USE_SQ_YOY:
        try:
            from income_idx import IncomeIdx
            sq_idx = IncomeIdx.load(code_pos)
            if sq_idx is not None:
                print('成长增速：单季度同比（聚宽口径）')
        except Exception as e:
            print('⚠️ 单季同比不可用（%s），回退累计同比' % str(e)[:60])
            sq_idx = None

    # PE(TTM) 索引：价值线要用「总市值 / TTM净利润」而不是「价格 / 单期EPS」
    #   两者相关系数只有 0.01~0.13，而价值线有「PE 8~50」硬过滤，
    #   口径错就等于换了一批股票。
    pe_idx = None
    try:
        from pe_idx import PeTtmIdx
        pe_idx = PeTtmIdx.load(code_pos)
        if pe_idx is not None:
            print('PE 口径：TTM 净利润（聚宽一致）')
    except Exception as e:
        print('⚠️ PE(TTM) 不可用（%s），回退价格/EPS' % str(e)[:60])
        pe_idx = None
    print('财务/分红索引 %.1fs' % (time.time() - t0))

    # 择时信号：> 0 判为进攻（red），< 0 判为防守（green）
    if TIMING == 'none':
        hist = None                      # select 里 hist is None → 永远 red
        print('择时：关闭（永远进攻）')
    elif TIMING in ('bond', 'bond_inv'):
        _z = bond_signal(dates)
        hist = (-_z if TIMING == 'bond' else _z) if _z is not None else None
        print('择时：股债信号（%s）' % ('广发方向：z高→防守' if TIMING == 'bond'
                                      else '中金方向：z高→进攻'))
    else:
        hist = macd_monthly(px[BENCH]) if BENCH in px.columns else None
        print('择时：沪深300 月线 MACD')

    bt_mask = (dates >= BT_START) & (dates <= BT_END)
    bt_rows = np.flatnonzero(bt_mask)
    bt_dates = dates[bt_rows]
    rebal = month_first_days(bt_dates, REBAL_FREQ)
    total_reb = len(rebal)
    progress.start(total_reb, params={
        '价格口径': '前复权' if ADJ else '不复权+分红',
        '策略': getattr(STRATEGY, 'name', type(STRATEGY).__name__),
        '区间': '%s ~ %s' % (BT_START, BT_END)})

    # 除权除息表：无论哪种价格口径都要建，因为**红利税是真金白银要交的**。
    #   ADJ=False（不复权）：现金分红入账 + 送转股调整股数 + 配股扣现金
    #   ADJ=True（前复权）：价格序列已含分红和送转，只扣税、不动股数
    #
    # ⚠️ 为什么两种口径处理不同：
    #   前复权把「10 送 10」折算成历史价腰斩，股数保持不变；
    #   不复权保留真实价腰斩，必须同时把股数翻倍。
    #   两个都做就会重复计算，净值凭空翻倍。
    div_map = {}
    for c, ex, dps, sb, ar, ap in zip(div['code'], div['ex_date'],
                                      div['dps'], div['share_ratio'],
                                      div['allot_ratio'], div['allot_price']):
        if (dps and dps > 0) or (sb and sb > 0):
            div_map.setdefault(pd.Timestamp(ex), []).append(
                (c, float(dps or 0), float(sb or 0),
                 float(ar or 0), float(ap or 0)))

    holding = OrderedDict()
    cash = CAPITAL
    nav_rows, trades, picks = [], [], []
    # 每次 run 都要重置这些全局状态，否则同进程连跑多次会串味
    global _DD_TRIG, _DD_LOW, _DD_N, _VOL_N_ADJ, _VOL_EXPO, _MA_ARR, _MA_N
    _DD_TRIG, _DD_LOW, _DD_N, _VOL_N_ADJ, _VOL_EXPO = False, 1.0, 0, 0, 1.0
    _MA_N = 0

    # 读 LPPL 泡沫信号（已 shift(1)，文件不存在就降级为不启用）
    global _LPPL_ARR
    _LPPL_ARR = None
    if LPPL_TIMING:
        _lp = os.path.join(BASE, 'lppl_signal.parquet')
        if os.path.exists(_lp):
            _d = pd.read_parquet(_lp)['sig']
            _d.index = pd.to_datetime(_d.index)
            _LPPL_ARR = _d.reindex(dates, method='ffill').fillna(False).values.astype(bool)
            print('LPPL 泡沫信号：%d 个交易日触发（%.1f%%）'
                  % (_LPPL_ARR.sum(), _LPPL_ARR.mean() * 100))
        else:
            print('LPPL 信号缺失（先跑 lppl_backtest.py），已关闭')

    # 预计算双均线信号（shift(1)：今天收盘算的信号，明天才用）
    if MA_TIMING:
        _b = px[BENCH_CMP].reindex(dates).ffill() if BENCH_CMP in px.columns else None
        if _b is not None and _b.notna().sum() > MA_SLOW:
            _mf = _b.rolling(MA_FAST).mean()
            _ms = _b.rolling(MA_SLOW).mean()
            _sig = ((_b > _mf) & (_b > _ms)).shift(1).fillna(True)
            _MA_ARR = _sig.values.astype(bool)
            print('双均线择时：%s close>MA%d 且 >MA%d ｜ 满仓占比 %.0f%%'
                  % (BENCH_CMP, MA_FAST, MA_SLOW, _MA_ARR.mean() * 100))
        else:
            _MA_ARR = None
            print('双均线择时：基准数据不足，已降级为满仓')
    else:
        _MA_ARR = None

    date2row = {d: i for i, d in enumerate(dates)}
    buy_date = {}                      # code -> 建仓日，红利税按持股时长分档

    # 涨停开板检查所需（日线近似）
    up_mat = sealed = None
    if LIMIT_OPEN_CHECK and PR is not None:
        up_mat = limit_up_matrix(PR, dates, codes)
        sealed = PR >= up_mat - 0.011
        print('涨停开板检查：开启（日线近似，用开盘价当 11:00）')
    target_pool = set()

    for d in bt_dates:
        row = date2row[d]
        for c, dps, sb, ar, ap in div_map.get(pd.Timestamp(d), []):
            if c not in holding or holding[c] <= 0:
                continue

            # ① 现金分红：以「除权前股数」为基数
            if dps > 0:
                amt = holding[c] * dps
                tax = amt * div_tax_rate(c, d, buy_date)
                net = amt - tax
                if ADJ:
                    # 前复权：分红已折进价格（价格不跌，相当于自动再投资）。
                    #
                    # ⚠️ 这里曾经扣了红利税，是【重复扣税】的 bug。
                    #    实测（verify_adj_tax.py，1421 个除权日）：
                    #        中位 0.9966，88.7% 的样本落在 K≈1.00
                    #    → xtdata 的 front_ratio 是【税前】复权（全额分红已折进价格）
                    #    → 再扣一次税等于扣两遍，按 3%×10% 估算低估约 0.3pp/年。
                    #
                    # 口径定义（务必写进报告）：
                    #    ADJ=True  → 税前总收益，用于和指数/外部基准对比
                    #    ADJ=False → 税后实际收益，用于实盘预期
                    pass
                elif DIV_REINVEST:
                    # **红利再投资**：除权日当天按不复权收盘价立即买入同一只股票。
                    #
                    # CRSP 的标准总收益口径就是这么算的：
                    #     再投资因子 = 1 + D / P0（D=分红，P0=除权前价）
                    #     总收益水平 = 除权后价 × 因子
                    # 等价于「把分红按除权价折成新股」。
                    #
                    # 之前是 cash += net，要等下次月度调仓才买 —— 现金最多闲置 20 个交易日，
                    # 这既不符合「红利再投资」的定义，也是「不复权 11.54% vs 前复权 11.08%」
                    # 这个差异的主要来源之一（前复权是自动立即再投资的）。
                    pc = P[row, jj] if (jj := code_pos.get(c)) is not None else np.nan
                    if np.isfinite(pc) and pc > 0:
                        holding[c] += net / pc
                    else:
                        cash += net          # 拿不到当日价才退回现金
                else:
                    cash += net

            # ② 送转股：只有不复权口径才调整股数
            #    前复权的价格序列已经按送转比例折算过（10送10 → 历史价腰斩），
            #    股数再翻倍就是重复计算。
            if (not ADJ) and sb > 0:
                holding[c] = holding[c] * (1.0 + sb)

            # ③ 配股：要额外掏现金，钱不够就放弃（现实中可以部分放弃）
            if (not ADJ) and ar > 0 and ap > 0:
                need = holding[c] * ar * ap
                if cash >= need:
                    cash -= need
                    holding[c] = holding[c] * (1.0 + ar)

        # 每日开板检查（在调仓之前）
        if LIMIT_OPEN_CHECK and up_mat is not None and holding:
            holding, cash = check_limit_open(row, d, holding, cash, O, up_mat,
                                             sealed, code_pos, target_pool, trades)

        # ---- 每日敞口检查（不等月度调仓）----
        # 2015-06 的教训：波动是 6 月中下旬才飙的，只挂在调仓日要等到 7 月初，
        # 最惨的那一周已经跌完。所以每天用「截至昨天的已实现净值」算一次波动，
        # 偏差超过 VOL_BAND 就按当日开盘价调仓（T+1 执行，无未来函数）。
        _need_adj = ((VOL_TARGET > 0 and VOL_DAILY) or DD_STOP > 0
                     or (MA_TIMING and _MA_ARR is not None)
                     or (LPPL_TIMING and _LPPL_ARR is not None))
        if _need_adj and holding and row > 0:
            _e = 1.0
            # LPPL 泡沫信号（已 shift(1)）
            if LPPL_TIMING and _LPPL_ARR is not None:
                _e = min(_e, LPPL_EXPO if _LPPL_ARR[row] else 1.0)
            # 双均线仓位信号（已 shift(1)，无未来函数）
            if MA_TIMING and _MA_ARR is not None:
                _e = min(_e, 1.0 if _MA_ARR[row] else MA_EXPO)
            if VOL_TARGET > 0 and VOL_DAILY and len(nav_rows) > VOL_WIN:
                _vals = np.array([x[1] for x in nav_rows[-VOL_WIN - 1:]], dtype=float)
                _vals = _vals[_vals > 0]
                if len(_vals) > 20:
                    _lr = np.diff(np.log(_vals))
                    _rv = float(np.std(_lr, ddof=1)) * np.sqrt(252)
                    if _rv > 1e-6:
                        _e = float(np.clip(VOL_TARGET / _rv, VOL_EXPO_MIN, 1.0))
            # 回撤止损：从历史高点回撤超阈值就降仓，反弹够才回场
            if DD_STOP > 0 and nav_rows:
                _nav_now = float(nav_rows[-1][1])
                _peak = max(float(x[1]) for x in nav_rows)
                _dd_now = _nav_now / _peak - 1 if _peak > 0 else 0.0
                if not _DD_TRIG:
                    if _dd_now <= -DD_STOP:
                        _DD_TRIG = True
                        _DD_LOW = _nav_now
                        _DD_N += 1
                    _e = min(_e, 1.0)
                else:
                    _DD_LOW = min(_DD_LOW, _nav_now)
                    if _DD_LOW > 0 and _nav_now / _DD_LOW - 1 >= DD_RESUME:
                        _DD_TRIG = False
                        _e = min(_e, 1.0)
                    else:
                        _e = min(_e, DD_EXPO)
            _VOL_EXPO = _e
            # 均线信号变化快（13 年 ~300 次），用更小的 band 才跟得上；
            # LPPL 一年只触发一两次，band 可以放宽
            if LPPL_TIMING and _LPPL_ARR is not None:
                _band = LPPL_BAND
            elif MA_TIMING and _MA_ARR is not None:
                _band = MA_BAND
            else:
                _band = VOL_BAND
            holding, cash, _adj = adjust_exposure(
                holding, cash, O[row], code_pos, trades, d, row, AM,
                _e, _band)
            if _adj:
                _VOL_N_ADJ += 1
                _MA_N += 1

        is_reb = d in rebal
        if is_reb:
            # 目标波动率缩放：用「已经实现」的净值序列算波动（无未来函数）。
            # nav_rows 只包含今天之前的数据，因为本条在 append 之前。
            # （全局声明已在上面每日检查处写过，这里不重复）
            if VOL_TARGET > 0 and len(nav_rows) > VOL_WIN:
                vals = np.array([x[1] for x in nav_rows[-VOL_WIN - 1:]], dtype=float)
                vals = vals[vals > 0]
                if len(vals) > 20:
                    lr = np.diff(np.log(vals))
                    rv = float(np.std(lr, ddof=1)) * np.sqrt(252)
                    _VOL_EXPO = (float(np.clip(VOL_TARGET / rv, VOL_EXPO_MIN, 1.0))
                                 if rv > 1e-6 else 1.0)
                else:
                    _VOL_EXPO = 1.0
            else:
                _VOL_EXPO = 1.0

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
                      sq_idx=sq_idx, pe_idx=pe_idx,
                      cfg=CONFIG)
            target, style = STRATEGY.select(ctx)

            # 选股注入：用外部目标池覆盖本地选股。
            #   · 只在 PICK_OVERRIDE 非 None 时生效（默认关闭）
            #   · 日期按「最近匹配」：本地 sdate 是 T-EXEC_LAG（通常 T-1），
            #     而外部记录的是成交日 T，直接等值匹配会全部落空
            if PICK_OVERRIDE:
                _ov = _match_override(sdate)
                if _ov is not None:
                    # 必须过滤掉本地没有行情的代码！
                    #   聚宽日志里有已退市/被并购的股票（如 000038.SZ 深大通），
                    #   本地的 daily 里根本没有它，直接传给 rebalance 会
                    #   KeyError: '000038.SZ'。
                    #   注入的目的是「验证框架」，不是「完整复现聚宽持仓」，
                    #   缺数据的股票只能跳过（实盘同样买不到）。
                    target = [c for c in _ov if c in code_pos]
                    _miss = len(_ov) - len(target)
                    if _miss and not globals().get('_OV_WARNED'):
                        globals()['_OV_WARNED'] = True
                        print('  ⚠️ 注入池里有 %d 只本地无行情（已跳过）' % _miss)
            picks.append((str(d.date())[:10], ','.join(target), style))
            target_pool = set(target)
            # 昨日封板的持仓股不参与月度卖出，留给次日开板检查
            skip = set()
            if LIMIT_OPEN_CHECK and sealed is not None and row > 0:
                skip = {c for c in holding
                        if code_pos.get(c) is not None and sealed[row - 1, code_pos[c]]}
            holding, cash = rebalance(row, target, holding, cash, O, code_pos,
                                      trades, d, skip_sell=skip, AM=AM,
                                      O_RAW=O_RAW, UP=UP_M, DN=DN_M)
            # 维护建仓日：新进池的记当天，已清仓的删掉
            for c in holding:
                if c not in buy_date:
                    buy_date[c] = pd.Timestamp(d)
            for c in list(buy_date):
                if c not in holding:
                    del buy_date[c]

        mv = 0.0
        prow = P[row]
        for c, sh in holding.items():
            p = prow[code_pos[c]]
            if np.isfinite(p):
                mv += sh * p
        nav_rows.append((d, cash + mv, cash, len(holding), _VOL_EXPO))

        if is_reb:
            progress.step(len(picks), total_reb, str(d.date())[:10],
                          nav_point=(str(d.date())[:10], (cash + mv) / CAPITAL),
                          pick=(str(d.date())[:10], ','.join(target), style),
                          trade=(trades[-1] if trades else None))

    nav = pd.DataFrame(nav_rows,
                       columns=['date', 'nav', 'cash', 'n_hold', 'expo']
                       ).set_index('date')

    # 把本轮的「每次调仓目标池」挂到模块级变量上。
    # 用途：对齐验证脚本要拿 picks 和聚宽的真实持仓逐期比对，
    #       而 run() 只返回 nav/trades，picks 会随局部作用域消失。
    global LAST_PICKS
    LAST_PICKS = list(picks)

    if SAVE:
        pd.DataFrame(picks, columns=['date', 'codes', 'style']).to_parquet(
            os.path.join(BASE, 'picks.parquet'), index=False)
        save_meta()
        save_run(nav, trades, picks, time.time() - t0)
    # 自动推前端：不管谁调 run()（脚本 / -c / __main__），都会更新 progress.json
    _publish(nav, trades, time.time() - t0)
    return nav, trades


def load_bench_cmp(dates=None):
    """业绩对比基准的收盘序列。

    注意和 BENCH 的区别：
      BENCH      = 沪深300，用于算 Beta 和 MACD 风格切换（市场代理，标准做法）
      BENCH_CMP  = 上证红利，只用于业绩对比（红利策略该和红利指数比）
    两者分开，换对比基准不会改动选股逻辑。
    """
    px = load_bars()['px']
    if BENCH_CMP not in px.columns:
        return None
    s = px[BENCH_CMP].dropna()
    return s.loc[dates[0]:dates[-1]] if dates is not None else s


def _publish(nav, trades, secs):
    """把本次结果推到前端（web/progress.json）。

    放在 run() 末尾自动调用，而**不是**只在 __main__ 里调 ——
    否则任何直接 `B.run()` 的脚本（单区间回测、参数扫描）都不会更新前端，
    表现为「回测跑完了，前端还是上一次的数据」。这个坑已经踩过多次：
      · 为了快，用 python -c "B.run()" 跑
      · 结果 progress.json 停在几十分钟前，用户以为前端坏了
    progress.ENABLED=False 时自动跳过（参数扫描用）。
    """
    if not progress.ENABLED:
        return None
    n = nav['nav']
    bench = load_bench_cmp([n.index[0], n.index[-1]])
    st = compute_stats(nav, trades, bench)
    st['mode'] = MODE_LABEL
    st['tag'] = RUN_TAG          # 前端标题栏显示「这是什么回测」
    st['secs'] = round(secs, 1)

    yearly, prev = [], CAPITAL
    for k, v in n.resample('YE').last().items():
        yearly.append([str(k.year), round(float(v / prev - 1) * 100, 2)])
        prev = v

    bc = []
    if bench is not None and len(bench) > 1:
        b = bench.reindex(n.index).ffill()
        base_b = b.dropna().iloc[0]
        bc = [[str(i.date()), round(float(v / base_b), 4)]
              for i, v in b.items() if np.isfinite(v)]

    # 全量净值 + 回撤序列（progress.step 每 20 天才记一点，会漏掉最深的那天）
    # ⚠️ 必须归一化到 1.00：nav 列的单位是「元」（100 万 -> 1178 万），
    #    直接传原值会让前端把 263 万当成净值，Y 轴刻度和悬停全变成天文数字。
    _n0 = float(n.iloc[0]) or 1.0
    _navpts = [[str(i.date()), round(float(v) / _n0, 4)] for i, v in n.items()]
    _ddser = (n / n.cummax() - 1) * 100
    _ddpts = [round(float(v), 2) for v in _ddser.values]
    progress.set_factors(factor_status())      # 前端「因子库」面板
    progress.set_pipeline(load_pipeline())     # 前端「数据清洗」面板
    progress.finish(st, yearly, bc, nav=_navpts, dd=_ddpts)
    return st


def _round_trips(trades):
    """从成交流水重建每笔持仓的盈亏，用来算胜率和盈亏比。"""
    if not trades:
        return np.array([])
    tr = pd.DataFrame(trades, columns=['date', 'code', 'side', 'shares', 'price'])
    tr['date'] = pd.to_datetime(tr['date'])
    pos, pnl = {}, []
    for _, x in tr.iterrows():
        if x['side'] == 'buy':
            pos.setdefault(x['code'], []).append([x['date'], float(x['shares']), float(x['price'])])
        else:
            lst = pos.get(x['code'])
            if not lst:
                continue
            left = float(x['shares'])
            while left > 0 and lst:
                d0, s0, p0 = lst[0]
                use = min(left, s0)
                pnl.append((x['price'] - p0) * use)
                left -= use
                if use >= s0:
                    lst.pop(0)
                else:
                    lst[0][1] = s0 - use
    return np.array(pnl)


def _top_drawdowns(n, k=3):
    """找出前 k 个「互不重叠」的回撤段。

    ⚠️ 不能用「反复取 dd 最低点 + 局部清零」的写法：
    2015-06 到 2016-02 净值一直没回到前高，属于**同一段**，
    但局部清零后第二次的 idxmin 还会落在这一段里（只是晚几天），
    结果前 3 名全是同一段（-31.71 / -30.58 / -30.08）。

    正确做法：按「水下区间」分段 —— 净值 < 前高 的连续区间算一段，
    每段取最深点。2015-06~2016-02、2018 全年、2024-10 各自独立。
    """
    peak = n.cummax()
    dd = (n / peak - 1) * 100
    uw = (dd < -0.5).values          # 水下 0.5% 以上才算
    idx = n.index
    out = []

    i = 0
    L = len(uw)
    while i < L:
        if not uw[i]:
            i += 1
            continue
        j = i
        while j + 1 < L and uw[j + 1]:
            j += 1
        seg = dd.iloc[i:j + 1]
        t = seg.idxmin()
        if np.isfinite(dd.loc[t]) and dd.loc[t] < -1.0:
            p = n.loc[:idx[i]].idxmax()
            out.append({
                'start': str(p.date()),
                'end': str(t.date()),
                'pct': round(float(dd.loc[t]), 2),
                'peak': round(float(n.loc[p]), 4),
                'trough': round(float(n.loc[t]), 4),
                'days': int((t - p).days),
            })
        i = j + 1

    out.sort(key=lambda x: x['pct'])      # 最深的排前面
    return out[:k]


def compute_stats(nav, trades, bench=None):
    """统一的绩效口径，供 history 快照与前端使用。

    基准相关指标（超额/α/β/信息比率）只在传入 bench 时计算。
    """
    n = nav['nav']
    # ⚠️ base 必须用「初始投入 CAPITAL」，不能用 n.iloc[0]。
    #    n.iloc[0] 是「第一天收盘、已建仓且已扣完交易成本」的净值，
    #    用它当基准等于把首日建仓成本从收益里抹掉，累计收益会虚高。
    #    实测（2016-01 起）：用 n.iloc[0] 得 256.63% / 年化 12.45%，
    #    用 CAPITAL 得 228.85% / 年化 11.61% —— 差 7.8pp 累计、0.84pp 年化。
    #    report() 一直用的是 CAPITAL，所以日志打 11.61%、
    #    前端却显示 12.45%，两边对不上就是这里造成的。
    #    归一化过的 nav（起点 1.0）也兼容：此时 n.iloc[0]≈1，直接退回它。
    if n.iloc[0] > 0.5 * CAPITAL:
        base = float(CAPITAL)
    else:
        base = float(n.iloc[0])
    years = (n.index[-1] - n.index[0]).days / 365.25
    total = n.iloc[-1] / base - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dd = (n / n.cummax() - 1).min()
    r = n.pct_change().dropna()
    vol = r.std() * np.sqrt(252)
    ANN = 252
    RF = 0.02

    # 最大回撤区间（前端要在净值曲线上高亮这一段）
    _dds = n / n.cummax() - 1
    _trough = _dds.idxmin()
    _peak = n.loc[:_trough].idxmax()

    out = {
        'total_return': round(total * 100, 2),
        'cagr': round(cagr * 100, 2),
        'max_dd': round(dd * 100, 2),
        'calmar': round(cagr / abs(dd), 2) if dd < 0 else None,
        # 最大回撤的起止日期 + 峰值净值，前端用来在曲线上画阴影
        'mdd_start': str(_peak.date()),
        'mdd_end': str(_trough.date()),
        'mdd_peak': round(float(n.loc[_peak]), 4),
        'mdd_trough': round(float(n.loc[_trough]), 4),
        'mdd_days': int((_trough - _peak).days),
        # 前 3 大回撤（互不重叠），前端按深浅分三档画
        'dds': _top_drawdowns(n, 3),
        'sharpe': round((cagr - RF) / vol, 3) if vol > 0 else None,
        'sortino': None,
        'vol': round(vol * 100, 2),
        'trades': len(trades),
        'final': round(float(n.iloc[-1]), 0),
        'years': round(years, 2),
        'win_rate': None, 'pl_ratio': None, 'n_win': 0, 'n_lose': 0,
        'bench': None, 'excess': None, 'alpha': None, 'beta': None,
        'ir': None, 'ex_dd': None, 'day_win': round((r > 0).mean(), 3),
    }

    # 索提诺：用下行波动
    down = r[r < 0].std() * np.sqrt(ANN)
    if down and np.isfinite(down) and down > 0:
        out['sortino'] = round((cagr - RF) / down, 3)

    # 按持仓周期算胜率与盈亏比
    pn = _round_trips(trades)
    if len(pn):
        win = int((pn > 0).sum())
        lose = int((pn < 0).sum())
        out['n_win'], out['n_lose'] = win, lose
        if win + lose:
            out['win_rate'] = round(win / (win + lose), 3)
        if win and lose:
            out['pl_ratio'] = round(pn[pn > 0].mean() / abs(pn[pn < 0].mean()), 3)

    # 基准对比
    if bench is not None and len(bench) > 10:
        b = bench.reindex(n.index).ffill()
        b = b.loc[n.index[0]:].dropna()
        if len(b) > 10:
            btot = b.iloc[-1] / b.iloc[0] - 1
            br = b.pct_change().dropna()
            idx = r.index.intersection(br.index)
            rs, bs = r.loc[idx], br.loc[idx]
            out['bench'] = round(btot * 100, 2)
            out['excess'] = round(((1 + total) / (1 + btot) - 1) * 100, 2)
            if len(rs) > 10 and np.var(bs, ddof=1) > 0:
                beta = np.cov(rs, bs)[0, 1] / np.var(bs, ddof=1)
                bcagr = (1 + btot) ** (1 / years) - 1
                out['beta'] = round(beta, 3)
                out['alpha'] = round(cagr - beta * bcagr - (1 - beta) * RF, 3)
                ex = (1 + rs) / (1 + bs) - 1
                ex_tot = (1 + total) / (1 + btot) - 1
                ex_cagr = (1 + ex_tot) ** (1 / years) - 1 if years > 0 else np.nan
                te = ex.std() * np.sqrt(ANN)
                if te > 0:
                    out['ir'] = round(ex_cagr / te, 3)
                exn = (1 + ex).cumprod()
                out['ex_dd'] = round((exn / exn.cummax() - 1).min() * 100, 2)
    return out


def save_run(nav, trades, picks, secs):
    """把这次回测的完整结果落一份 JSON，供前端「历史回测」列表回看。

    只保留最近 MAX_RUNS 次，避免无限堆积。
    """
    import json
    d = os.path.join(BASE, 'runs')
    os.makedirs(d, exist_ok=True)
    rid = time.strftime('%Y%m%d_%H%M%S')
    n = nav['nav']
    yearly = []
    prev = n.iloc[0]
    for y, g in n.groupby(n.index.year):
        yearly.append([int(y), round((g.iloc[-1] / prev - 1) * 100, 2)])
        prev = g.iloc[-1]
    bench = load_bench_cmp([n.index[0], n.index[-1]])
    st = compute_stats(nav, trades, bench)
    st['mode'] = MODE_LABEL
    st['tag'] = RUN_TAG
    st['secs'] = round(secs, 1)
    bench_curve = []
    if bench is not None and len(bench) > 1:
        b = bench.reindex(n.index).ffill()
        base_b = b.dropna().iloc[0]
        bench_curve = [[str(i.date()), round(float(v / base_b), 4)]
                       for i, v in b.items() if np.isfinite(v)]
    obj = {
        'id': rid,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': current_config(),
        'stats': st,
        'yearly': yearly,
        # 日频净值（3600 点左右），前端画曲线用
        'nav': [[str(i.date()), round(float(v), 4)] for i, v in n.items()],
        # 对比基准（归一化到 1.00），前端叠一条灰线
        'bench': bench_curve,
        'picks': [[p[0], p[1], p[2]] for p in picks],
        'trades_tail': [[str(t[0])[:10], t[1], t[2], int(t[3]), round(float(t[4]), 3)]
                        for t in trades[-60:]],
    }
    try:
        # 同一秒内连跑多次时加序号，避免互相覆盖
        path = os.path.join(d, rid + '.json')
        k = 1
        while os.path.exists(path):
            k += 1
            path = os.path.join(d, '%s_%d.json' % (rid, k))
            obj['id'] = '%s_%d' % (rid, k)
        json.dump(obj, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
        fs = sorted(f for f in os.listdir(d) if f.endswith('.json'))
        for old in fs[:-MAX_RUNS]:
            os.remove(os.path.join(d, old))
    except Exception as e:
        print('保存历史快照失败：%s' % e)


_OV_ARR = None    # np.array(datetime64)，按日期升序
_OV_KEYS = None   # 对应的原始 key 列表


def load_jq_factors():
    """加载聚宽导出的因子表，供 select() 直接使用。

    返回 {date_str: DataFrame(index=code, columns=[dividend_yield, beta, pe_ratio, ...])}
    日期是「月末交易日」，和本地 sdate（信号日）应该能精确对上。
    """
    p = os.path.join(BASE, 'clean_pipe', 'jq_factors.parquet')
    if not os.path.exists(p):
        print('⚠️ 缺 clean_pipe/jq_factors.parquet，USE_JQ_FACTORS 无效')
        return None
    d = pd.read_parquet(p)
    d['date'] = pd.to_datetime(d['date'])
    out = {}
    for dt, g in d.groupby('date'):
        out[str(dt.date())] = g.set_index('code')
    print('聚宽因子表：%d 期 ｜ %s ~ %s'
          % (len(out), min(out).__str__(), max(out).__str__()))
    return out


def _jq_slice(sdate):
    """给信号日找最近的聚宽因子快照（±6 天容错）。"""
    if not _JQ_F:
        return None
    key = str(pd.Timestamp(sdate).date())
    if key in _JQ_F:
        return _JQ_F[key]
    best, gap = None, 999
    t = pd.Timestamp(sdate)
    for k in _JQ_F:
        g = abs((pd.Timestamp(k) - t).days)
        if g < gap:
            gap, best = g, k
    return _JQ_F[best] if gap <= 6 else None


def _match_override(sdate, tol_days=12):
    """给本地选股日找最近的「外部目标池」日期。

    ⚠️ 不能等值匹配：
       本地 sdate = 信号日（T - EXEC_LAG，通常 T-1），
       聚宽日志记的是成交日 T，直接比字符串全部落空。
       允许 ±12 天是因为月调仓遇到长假会错开（如 2016-10-10 vs 2016-10-08）。
    """
    global _OV_ARR, _OV_KEYS
    if _OV_ARR is None:
        ks = sorted(PICK_OVERRIDE.keys())
        _OV_KEYS = ks
        _OV_ARR = np.array([np.datetime64(k) for k in ks])
    if not len(_OV_ARR):
        return None
    d = np.datetime64(pd.Timestamp(sdate))
    gap = ( _OV_ARR - d).astype('timedelta64[D]').astype(int)
    j = int(np.abs(gap).argmin())
    if abs(gap[j]) > tol_days:
        return None
    return PICK_OVERRIDE[_OV_KEYS[j]]


def load_pipeline():
    """读数据清洗管线的汇总（clean_pipe/_index.json），推给前端。

    回测结果可不可信，第一层取决于数据干不干净。
    这块信息必须出现在前端，而不是只在终端里打一遍就没了。

    返回：[{name, overall, rows, dropped, dims, ts}, ...]
    没跑过管线时返回空列表，前端显示「未清洗」。
    """
    p = os.path.join(BASE, 'clean_pipe', '_index.json')
    if not os.path.exists(p):
        return []
    try:
        d = json.load(open(p, encoding='utf-8'))
    except Exception:
        return []
    out = []
    for r in d.get('datasets', []):
        if 'error' in r:
            out.append({'name': r.get('name', '?'), 'overall': 0,
                        'rows': 0, 'dropped': 0, 'dims': {},
                        'error': str(r['error'])[:80]})
            continue
        out.append({
            'name': r.get('name', '?'),
            'overall': r.get('overall'),
            'rows': r.get('rows', 0),
            'dropped': r.get('dropped', 0),
            'raw_rows': r.get('raw_rows', 0),
            'dims': r.get('dims', {}),
            'audit': r.get('audit_rows', 0),
            'secs': r.get('secs'),
            'ts': r.get('ts', ''),
        })
    return out




def save_meta():
    """把参数 + git commit 落到 nav_meta.json，和 nav.parquet 成对存在。"""
    import json
    import subprocess
    meta = current_config()
    try:
        r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd='..',
                           capture_output=True, text=True, timeout=8)
        d = subprocess.run(['git', 'status', '--porcelain'], cwd='..',
                           capture_output=True, text=True, timeout=8)
        meta['git'] = {'commit': r.stdout.strip() or 'unknown',
                       'dirty': bool(d.stdout.strip())}
    except Exception:
        meta['git'] = {'commit': 'unknown', 'dirty': None}
    try:
        json.dump(meta, open(os.path.join(BASE, 'nav_meta.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
    except Exception:
        pass


def div_tax_rate(code, d, buy_date):
    """A股差别化股息红利税：按持股期限分档。

      持股 ≤ 1 个月      20%
      1 个月 ~ 1 年       10%
      > 1 年              0%（免税）

    实务上是「卖出时按实际持股期补扣」，这里近似为「分红入账时按已持有天数扣」——
    因为我们的持仓绝大多数持有到下次调仓（≥1 个月），差异有限。
    """
    if not DIV_TAX:
        return 0.0
    bd = buy_date.get(code)
    if bd is None:
        return 0.10                     # 建仓日记不清，按中间档
    days = (d - bd).days
    if days > 365:
        return 0.0
    if days > 30:
        return 0.10
    return 0.20


def slip_bp(price, amt_yuan, trade_yuan):
    """本次交易的滑点（bp）。

    price     ：成交价（元）—— 决定 tick 成本占比
    amt_yuan  ：该股当日成交额（元）
    trade_yuan：本次委托金额（元）

    价差 = 1 个 tick 占股价的比例（3 元股 33bp、10 元股 10bp、30 元股 3.3bp），
    冲击 = 150bp × sqrt(参与率)，两者相加。
    """
    if not np.isfinite(price) or price <= 0:
        return SLIP_SPREAD_MAX + SLIP_IMPACT_K * np.sqrt(SLIP_CAP)
    tick_bp = SLIP_TICK / price * 1e4
    spread = min(max(tick_bp, SLIP_SPREAD_MIN), SLIP_SPREAD_MAX)
    if not np.isfinite(amt_yuan) or amt_yuan <= 0:
        return spread + SLIP_IMPACT_K * np.sqrt(SLIP_CAP)   # 缺成交额，按最差估
    part = min(trade_yuan / amt_yuan, SLIP_CAP)
    return spread + SLIP_IMPACT_K * np.sqrt(max(part, 0.0))


def adjust_exposure(holding, cash, o, code_pos, trades, d, row, AM, expo, band):
    """把股票总仓位调到 expo（双向：超配减、低配补）。

    抽成独立函数，是因为要在两个地方调用：
      · 月度调仓日（rebalance 开头）
      · 每日敞口检查（run 主循环）—— 只挂在调仓日的话，
        2015-06 波动飙升后要等到 7 月初才降仓，正好错过最惨的一周。

    band：偏差小于该值就不动，避免天天摩擦。
    返回 (holding, cash, 是否真的调整过)。
    """
    # 注意：这里不能写 `or expo <= 0` —— 那会把「清仓」这个动作本身挡掉
    # （expo=0 时直接 return，什么都不卖）。之前 LPPL 设 0% 完全没效果就是这个原因。
    if not holding:
        return holding, cash, False
    mv_now = 0.0
    for c, sh in holding.items():
        p = o[code_pos[c]]
        if np.isfinite(p) and p > 0:
            mv_now += sh * p
    total_now = cash + mv_now
    if total_now <= 0 or mv_now <= 0:
        return holding, cash, False
    cur_ratio = mv_now / total_now
    if abs(cur_ratio - expo) <= band:
        return holding, cash, False

    tgt_mv = total_now * expo
    if mv_now > tgt_mv:                       # 超配 → 按比例减
        keep = tgt_mv / mv_now
        for c in list(holding.keys()):
            j = code_pos[c]
            p = o[j]
            if not np.isfinite(p) or p <= 0:
                continue
            sh = holding[c]
            sell_sh = int(sh * (1 - keep) / 100) * 100
            if sell_sh <= 0:
                continue
            if sell_sh >= sh:
                sell_sh = sh
                holding.pop(c, None)
            else:
                holding[c] = sh - sell_sh
            if SLIP_ON and AM is not None:
                bp = slip_bp(p, AM[row, j], sell_sh * p)
                p = p * (1 - bp / 1e4)
            amt = sell_sh * p
            cash += amt - (max(5.0, amt * (COMMISSION + TRANSFER_FEE)) + amt * STAMP_TAX)
            trades.append((d, c, 'sell', sell_sh, p))
    else:                                     # 低配 → 按市值比例补
        need = min(tgt_mv - mv_now, cash * 0.98)
        if need > 0:
            for c in list(holding.keys()):
                j = code_pos[c]
                p = o[j]
                if not np.isfinite(p) or p <= 0:
                    continue
                w = holding[c] * p / mv_now
                buy_amt = need * w
                if buy_amt < p * 100:
                    continue
                buy_sh = int(buy_amt / p / 100) * 100
                if buy_sh <= 0:
                    continue
                if SLIP_ON and AM is not None:
                    bp = slip_bp(p, AM[row, j], buy_sh * p)
                    p = p * (1 + bp / 1e4)
                amt = buy_sh * p
                fee = max(5.0, amt * (COMMISSION + TRANSFER_FEE))
                if amt + fee > cash:
                    continue
                holding[c] += buy_sh
                cash -= amt + fee
                trades.append((d, c, 'buy', buy_sh, p))
    return holding, cash, True


def rebalance(row, target, holding, cash, O, code_pos, trades, d,
              skip_sell=(), AM=None, O_RAW=None, UP=None, DN=None):
    """skip_sell：昨日封板的持仓股，月度调仓时先不卖，留给次日的开板检查处理。

    这是原策略的关键设计（「昨日涨停股不列入月度卖出清单」）——
    因为它赌涨停次日还有冲高，等 11:00 开板再卖。

    O_RAW / UP / DN：涨跌停检查（LIMIT_CHECK=True 时由 run 传入）。
        O_RAW 是【不复权】开盘价（涨跌停价按真实价格定，不能和前复权价比）
        UP / DN 是当日涨停价 / 跌停价
        检查规则：
            开盘价 >= 涨停价 - 1e-6  → 买不进，跳过
            开盘价 <= 跌停价 + 1e-6  → 卖不出，跳过
    """
    o = O[row]
    _oraw = O_RAW[row] if O_RAW is not None else None
    _up = UP[row] if UP is not None else None
    _dn = DN[row] if DN is not None else None

    def _can_buy(j):
        """开盘就封涨停 → 买不进"""
        if _oraw is None or _up is None:
            return True
        p, u = _oraw[j], _up[j]
        if not (np.isfinite(p) and np.isfinite(u) and u > 0):
            return True
        return p < u - 1e-6

    def _can_sell(j):
        """开盘就封跌停 → 卖不出"""
        if _oraw is None or _dn is None:
            return True
        p, dn_ = _oraw[j], _dn[j]
        if not (np.isfinite(p) and np.isfinite(dn_) and dn_ > 0):
            return True
        return p > dn_ + 1e-6

    # ---- 目标波动率缩放：把总仓位调到 _VOL_EXPO（月频调仓日这一趟）----
    # 必须放在「成分不变就不动」的 early return 之前，
    # 否则目标池没变的月份永远不会调仓，这个功能等于没接。
    if VOL_TARGET > 0:
        holding, cash, _ = adjust_exposure(holding, cash, o, code_pos, trades,
                                           d, row, AM, _VOL_EXPO, 0.05)

    # 成分不变就不动：聚宽版只在「股票进出目标池」时交易，
    # 不做每月等权再平衡。原来每月剃一遍会多出近 200 笔无效交易
    # （每笔都交佣金+印花税），这也是回撤比聚宽大 3pp 的原因之一。
    cur_names = set(c for c, sh in holding.items() if sh > 0)
    if cur_names == set(target):
        return holding, cash

    for c in list(holding.keys()):
        if c in target or c in skip_sell:
            continue
        j = code_pos[c]
        p = o[j]
        if not np.isfinite(p) or p <= 0:
            continue
        if not _can_sell(j):
            continue        # 跌停封死，卖不出
        sh = holding.pop(c)
        # 卖出滑点：成交价下调（滑点按本次卖出金额估）
        if SLIP_ON and AM is not None:
            bp = slip_bp(p, AM[row, j], sh * p)
            p = p * (1 - bp / 1e4)
        amt = sh * p
        cash += amt - (max(5.0, amt * (COMMISSION + TRANSFER_FEE)) + amt * STAMP_TAX)
        trades.append((d, c, 'sell', sh, p))

    if not target:
        return holding, cash

    # 只买「新进池且当前没有」的票，钱平摊给它们；已持有的不加不减
    missing = [c for c in target if holding.get(c, 0) <= 0]
    if not missing:
        return holding, cash
    #
    # 权重必须在 **target 全集** 上归一化，再折算到 missing 子集。
    # 第一版写错了——直接在 missing 上归一化，于是「12 只里只换 3 只」时
    # 每只 missing 的权重变成 1/3=0.33，撞上 15% 上限后被截断，
    # 总权重不足 1 → 大量现金闲置 → 等权+上限本该无变化却掉了 326pp。
    n_tgt = max(len(target), 1)
    full = {}
    for c in target:
        w = _WEIGHTS.get(c, 0.0) if _WEIGHTS else 1.0
        full[c] = max(w, 0.0)
    s = sum(full.values())
    full = ({c: v / s for c, v in full.items()} if s > 0
            else {c: 1.0 / n_tgt for c in target})

    if MAX_WEIGHT > 0:
        for _ in range(6):
            over = [c for c in full if full[c] > MAX_WEIGHT + 1e-9]
            if not over:
                break
            for c in over:
                full[c] = MAX_WEIGHT
            others = [c for c in full if c not in over]
            rest = 1.0 - MAX_WEIGHT * len(over)
            if not others or rest <= 0:
                break
            tot_o = sum(full[c] for c in others)
            if tot_o <= 0:
                break
            for c in others:
                full[c] = full[c] / tot_o * rest

    # 折算到本次要买的那几只：按它们在全集里的相对权重分剩余现金
    s_miss = sum(full.get(c, 0.0) for c in missing)
    if s_miss > 0:
        ws = {c: full.get(c, 0.0) / s_miss for c in missing}
    else:
        ws = {c: 1.0 / len(missing) for c in missing}

    cash_pool = cash                      # 预算基数固定，不随循环内 cash 递减
    for c in missing:
        j = code_pos[c]
        p = o[j]
        if not np.isfinite(p) or p <= 0:
            continue
        if not _can_buy(j):
            continue        # 涨停封死，买不进
        budget = cash_pool * ws[c]
        # 买入滑点：成交价上调（用计划金额估参与率，金额与价格的循环依赖忽略不计）
        if SLIP_ON and AM is not None:
            bp = slip_bp(p, AM[row, j], budget)
            p = p * (1 + bp / 1e4)
        sh = int(budget * 0.98 / p / 100) * 100
        if sh <= 0:
            continue
        cost = sh * p
        fee = max(5.0, cost * (COMMISSION + TRANSFER_FEE))
        if cost + fee > cash:
            sh = int(cash * 0.97 / p / 100) * 100
            if sh <= 0:
                continue
            cost = sh * p
            fee = max(5.0, cost * (COMMISSION + TRANSFER_FEE))
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
    bench = load_bench_cmp([n.index[0], n.index[-1]])
    st = compute_stats(nav, trades, bench)

    print('=' * 66)
    print('区间 %s ~ %s（%.2f 年）｜ 口径 %s'
          % (n.index[0].date(), n.index[-1].date(), years, MODE_LABEL))
    print('期末 %.0f ｜ 累计 %.2f%% ｜ 年化 %.2f%% ｜ 最大回撤 %.2f%%'
          % (n.iloc[-1], total * 100, cagr * 100, dd * 100))
    print('交易 %d 笔 ｜ 总耗时 %.1fs' % (len(trades), secs))
    if st.get('bench') is not None:
        print('-' * 66)
        print('基准 %s ｜ 收益 %+.2f%% ｜ 超额 %+.2f%%'
              % (BENCH_CMP, st['bench'], st['excess']))
        print('α %s ｜ β %s ｜ 夏普 %s ｜ 索提诺 %s ｜ 信息比率 %s'
              % (st['alpha'], st['beta'], st['sharpe'], st['sortino'], st['ir']))
        print('胜率 %s（%d胜/%d负）｜ 盈亏比 %s ｜ 日胜率 %s'
              % (st['win_rate'], st['n_win'], st['n_lose'], st['pl_ratio'], st['day_win']))
        print('超额最大回撤 %+.2f%% ｜ 波动率 %.1f%%'
              % (st['ex_dd'] or 0, st['vol']))
    print('=' * 66)
    prev = CAPITAL
    for k, v in n.resample('YE').last().items():
        print('  %d: %+.1f%%' % (k.year, (v / prev - 1) * 100))
        prev = v
    # 直接返回 compute_stats 的全套指标（含 α/β/索提诺/信息比率/胜率/盈亏比），
    # 前端结果卡要用，不能只回几个基础字段
    st['mode'] = MODE_LABEL
    st['tag'] = RUN_TAG
    st['secs'] = round(secs, 1)
    return st


if __name__ == '__main__':
    # 支持任意区间，跑完自动推前端：
    #   python backtest_fast.py --begin 2022-11-07 --end 2026-11-06
    #   python backtest_fast.py --begin 2022-11-07        （只改起点）
    # 不加参数就是默认全区间。
    _argv = sys.argv
    if '--begin' in _argv:
        BT_START = _argv[_argv.index('--begin') + 1]
    if '--end' in _argv:
        BT_END = _argv[_argv.index('--end') + 1]

    t0 = time.time()
    # 崩了要显式推「失败」给前端。
    # 否则 progress.json 永远停在 start() 写的 running/加载数据，
    # 前端一直转圈，看起来像「前端坏了」，实际是回测挂了没人收尾。
    try:
        nav, trades = run()          # run() 内部已经 _publish 到前端
    except Exception as e:
        progress.fail('%s: %s' % (type(e).__name__, e))
        raise
    secs = time.time() - t0
    nav.to_parquet(os.path.join(BASE, 'nav.parquet'))
    pd.DataFrame(trades, columns=['date', 'code', 'side', 'shares', 'price']).to_parquet(
        os.path.join(BASE, 'trades.parquet'), index=False)
    report(nav, trades, secs)