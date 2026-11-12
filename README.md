# qbacktest · A股向量化回测框架

一个不绑定任何策略的本地回测引擎。框架只负责"与选股无关"的部分，选股逻辑通过插件接口注入。

- 单轮 5 年全市场回测 **3~5 秒**（5000 只 × 1200 交易日）
- 数据全部走本地 parquet，运行时不联网、不调 API
- 内置**数据指纹**与**运行记录**，保证结果可复现
- 自带前端看板（进度条 / 净值曲线 / 成交明细）

## 为什么快

三处热点全部前置计算，回测主循环里只做查表：

| 环节 | 常规写法 | 本框架 |
|---|---|---|
| Beta 面板 | 每个调仓日对候选股逐只算协方差 | `ret.rolling(win).cov(bench)` 一次算完全历史 |
| 财务/分红检索 | 每次对全表做布尔过滤 | `(code_id × 1e10 + yyyymmdd)` 拼成单调键 + `searchsorted` |
| 因子分位 | pandas 多重布尔索引 + quantile | numpy 掩码 + `argsort` |

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
| `beta_v / vol_v` | Beta 面板 / 波动率面板（窗口由 `--bt-win` 控制） |
| `fin_idx` | FN 字段 → KeyIndex，支持 `.latest(yyyymmdd)` |
| `div_idx` | 分红索引，支持 `.window_sum(lo, hi)` |
| `first_idx` | 每只股票首个有效行（判上市天数） |
| `hist` | 基准月线 MACD 柱值 |
| `hold` / `cfg` | 当前持仓 / 运行配置 |

## 数据契约

数据不入库。框架需要四个目录，全部 parquet，列名固定：

| 目录 | 必需列 | 说明 |
|---|---|---|
| `daily/` | `date, code, open, close, volume` | 前复权日线 |
| `daily_raw/` | `date, code, close` | 不复权日线（估值用） |
| `fin/` | `code, announce_time, FN1, FN96, FN107, FN238, …` | 财务，按公告日 |
| `div/` | `code, ex_date, dps` | 真实每股派现 |

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

1. 撮合按「当日开盘价」成交，不模拟滑点与冲击成本；需要更真实的撮合要自己改 `rebalance()`
2. 停牌用「成交量 > 0」近似
3. 股票池来自你提供的数据，本框架不做退市股补全

## 许可

MIT
