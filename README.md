# qbacktest · A股向量化回测框架

一个不绑定任何策略的本地回测引擎。框架只负责"与选股无关"的部分，选股逻辑通过插件接口注入。

- 单轮 **10 年**全市场回测 **10~15 秒**（5228 只 × 2600 交易日）
- 数据全部走本地 parquet，运行时不联网、不调 API
- 内置**数据指纹**与**运行记录**，保证结果可复现
- 自带前端看板（进度条 / 净值曲线 / 成交明细）

## 为什么快

三处热点全部前置计算，回测主循环里只做查表：

| 环节 | 常规写法 | 本框架 |
|---|---|---|
| Beta 面板 | 每个调仓日对候选股逐只算协方差 | 一次 `ewm(halflife=63)` 算完全历史 |
| 财务/分红检索 | 每次对全表做布尔过滤 | `(code_id × 1e10 + yyyymmdd)` 拼成单调键 + `searchsorted` |
| 因子分位 | pandas 多重布尔索引 + quantile | numpy 掩码 + `argsort` |

## 因子面板（2026-11 新增）

除了日线，框架预计算了三个「按公告日查询」的索引，策略通过 `ctx` 直接取：

| 索引 | 内容 | 口径 |
|---|---|---|
| `fin_idx` | FN 字段（EPS / 货币资金 / 净利润 / 总股本…） | 通达信 `fin`，按公告日 |
| `div_idx` | 每股派息（已按「此后所有送转」累积摊薄） | `dps_adj`，不是原始 `dps` |
| `sq_idx` | 单季同比三项：营收 / 净利润（含少数股东）/ 营业利润 | 累计差分 → 对齐常见财务库的「单季度」口径 |
| `pe_idx` | TTM 净利润 + 总股本 | PE = 总市值 / TTM 净利润 |

**关键：`div_idx` 必须用摊薄后的每股派息。** 原始 `dps` 是「除权前」口径，而算股息率时用的 `price` 是「除权后」价格，两者不在同一坐标系，10 送 20 会让结果虚高 3 倍。摊薄要按「此后所有送转」累积计算，不能只处理当期那一次（否则「先派息、后送转」会被漏掉）。

**关键：`pe_idx` 必须用 TTM 净利润。** 用「价格 / 单期累计 EPS」会系统性偏高 3~4 倍，而 PE 区间过滤通常是硬阈值，口径错就等于换了一批股票。

**关键：`sq_idx` 的净利润要用「含少数股东损益」那一列。** 归母净利润和含少数股东损益在建筑/地产/基建类公司能差 30%~50%，同比增速差异更大。营业利润则是完全独立的科目（含投资收益、公允价值变动），不能用净利润代替。

## 目录

```
qbacktest/
├── engine.py      数据加载 / 因子面板 / 回测主循环 / 撮合 / 绩效
├── strategy.py    策略接口（Ctx + Strategy 基类）
├── manifest.py    数据指纹 + 运行记录
└── progress.py    进度上报（写 web/progress.json，供前端轮询）
examples/
└── lowvol.py      示例策略：低波 + 大市值
run.py             CLI 入口
serve.py           本地 HTTP 服务（前端看板）
web/index.html     单文件前端
```

## 快速开始

```bash
pip install -r requirements.txt

# 数据放在与本仓库同级的 localdata/，或用环境变量指定
export QBACKTEST_DATA=/path/to/localdata

python run.py                          # 示例策略
python run.py --strategy examples.lowvol:LowVolLargeCap --raw
python run.py --start 2021-11-08 --end 2026-11-06 --bt-win 250
```

前端看板：

```bash
python serve.py              # 起在 8765，自动开浏览器
python run.py                # 另开一个窗口跑，页面实时刷新
```

## 写一个策略

只实现一个方法：

```python
from qbacktest.strategy import Strategy, Ctx, latest, safe_div

class MyStrategy(Strategy):
    name = 'my_strategy'

    def select(self, ctx: Ctx):
        price = ctx.P[ctx.row]
        alive = np.isfinite(price) & (price > 0)

        # 财务字段：FN1=基本EPS, FN96=归母净利, FN238=总股本 …
        asof = int(ctx.date.strftime('%Y%m%d'))
        eps = latest(ctx.fin_idx, 'FN1', asof, len(ctx.codes))

        # 不复权真实价（估值类因子必须用它）
        praw = ctx.PR[ctx.row]

        idx = np.flatnonzero(alive & np.isfinite(eps) & (eps > 0))
        order = idx[np.argsort(-eps[idx])][:20]
        return [ctx.codes[i] for i in order], 'my_style'
```

`ctx` 里已有的东西：

| 字段 | 说明 |
|---|---|
| `P / PR` | 收盘价矩阵：`P` 前复权，`PR` 不复权真实价 |
| `V / O` | 成交量 / 开盘价 |
| `beta_v / vol_v` | Beta 面板 / 波动率面板 |
| `fin_idx` | FN 字段 → KeyIndex，支持 `.latest(yyyymmdd)` |
| `div_idx` | 分红索引（已累积摊薄），支持 `.window_sum(lo, hi)` |
| `sq_idx` | 单季同比索引：`.latest(asof, codes)` → `(营收, 净利润, 营业利润)` |
| `pe_idx` | PE(TTM) 索引：`.latest(asof, codes)` → `(ttm净利润, 总股本)` |
| `first_idx` | 每只股票首个有效行（判上市天数） |
| `hist` | 基准月线 MACD 柱值 |
| `hold` / `cfg` | 当前持仓 / 运行配置 |

**Beta 口径**：`beta_v` 默认用「252 日 / 全 A 市值加权 / 指数加权 halflife=63 / 停牌日收益率填 0」，并对结果做「市值加权均值 + 等权 std + 2.5σ 截断」的标准化。这是常见风格因子库的口径。

## 数据契约

数据不入库。框架需要六个原始目录 + 一个清洗产物目录，全部 parquet，列名固定：

### 原始数据

| 目录 | 必需列 | 说明 |
|---|---|---|
| `daily/` | `date, code, open, close, volume` | 前复权日线 |
| `daily_raw/` | `date, code, close, volume` | 不复权日线（估值用；`volume>0` 判停牌） |
| `fin/` | `code, announce_time, FN1, FN96, FN107, FN238, …` | 财务，按公告日 |
| `div/` | `code, ex_date, Bonus, ShareBonus` | 每 10 股派息 / 送转 |
| `income/` | `code, report_date, revenue, net_profit_incl_min_int_inc, oper_profit` | 利润表 |
| `capital/` | `code, ann_date, total_capital, circulating_capital` | 历史股本（2010 起） |

### 清洗产物（`clean_pipe/`）

| 文件 | 内容 |
|---|---|
| `fin.parquet` | 清洗后财务 |
| `income.parquet` | 清洗后利润表（含 `net_profit` / `net_profit_incl` / `oper_profit`） |
| `div.parquet` | 清洗后分红（含 `dps` / `dps_adj`） |
| `capital.parquet` | 清洗后股本 |
| `income_yoy.parquet` | 单季同比三项 |
| `pe_ttm.parquet` | TTM 净利润 |

`code` 格式为 `600519.SH` / `000001.SZ`。

### 可复现

```bash
# 生成/更新数据指纹
python -m qbacktest.manifest --data ../localdata

# 校验手上的数据是不是同一份
python -m qbacktest.manifest --data ../localdata --check
```

指纹算法是"行数 + 关键列求和 + 分片文件名"的哈希，容忍 parquet 元数据差异，只比对数据本身。

每次运行的记录（`runs/` 下）包含：git commit、是否有未提交改动、完整配置、数据指纹、numpy/pandas 版本、绩效摘要。拿着这份记录，理论上可以逐字复现同一个结果。

## 依赖

Python 3.13，版本锁定在 `requirements.txt`：

```
numpy==2.5.4
pandas==3.1.1
pyarrow==26.0.1
```

## 已知局限

1. 撮合按「当日开盘价」成交；滑点与冲击成本需要自己实现（`examples/` 里有平方根冲击模型的参考实现）
2. 停牌用「成交量 > 0」近似。交易所口径的 `paused` 字段拿不到，一字板缩量场景会有误判
3. 股票池来自你提供的数据，本框架不做退市股补全 —— **存在幸存者偏差，回测结果偏乐观**
4. **Beta 无法精确复刻商业因子库**：它们对 A+H 公司按「A股×A价 + H股×H价×汇率 + 其余×每股净资产」计市值，
   而本地只有 `total_capital` / `circulating_capital`，拿不到 H 股分拆和汇率。
   实测排序相关 0.99（够用），但绝对值的离散度只有对方 1/2。
5. **财务字段按公告日 ffill**，同一天披露多期报表时只取最后一条（`aggfunc='last'`）。极少数公司在更正公告日会被漏掉。

## 许可

MIT
