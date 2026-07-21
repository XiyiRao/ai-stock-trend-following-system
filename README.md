# AI Stock Trend Following System

这是一个面向中际旭创（300308）的日频、纯多头趋势研究系统，按照《中际旭创_量化自动交易系统_完整实施操作指南》第4—12章实现数据质量、跨市场对齐、AI产业评分、个股评分、仓位风控、A股事件回测、稳健性研究和模拟盘日常流程。

> 本项目仅用于研究和教学，不构成投资建议。回测通过不代表未来盈利。完成至少20个真实交易日的模拟盘验收前，不应连接真实资金账户。

## 当前进度

| 章节 | 软件实现 | 仍需人工/时间完成 |
|---|---|---|
| 4. 环境、软件与账户 | 独立`.venv`、环境检查、安装脚本 | 创业板权限和券商模拟账户需本人向券商确认 |
| 5. 数据与质量控制 | Choice、Yahoo/缓存、OHLCV、交易日历、哈希和质量闸门 | 每次导出需确认Choice为前复权 |
| 6. 时差与未来函数 | 严格`US_Date < CN_Date`、滚动百分位、10日审计 | 节假日错位继续在模拟盘观察 |
| 7. AI产业状态 | AI股票、SOX、QQQ、美债四因子`AI_SCORE` | 参数上线前冻结 |
| 8. 个股因子 | 五维`STOCK_SCORE` | 参数上线前冻结 |
| 9. 信号、仓位与风控 | 仓位上限、回撤、MA120、ATR追踪止损、再平衡阈值 | 真实持仓以券商为准 |
| 10. A股回测 | T+1开盘、整手、涨跌停、停牌、滑点、佣金、印花税、限额 | 分钟成交模型属于后续升级 |
| 11. 回测评价与稳健性 | 基准比较、样本划分、走步、成本压力、参数扰动和验收闸门 | 结果仍需人工解释，不能只看收益率 |
| 12. 模拟盘与SOP | 持久化模拟账户、计划单、幂等、锁、日志、对账、停机开关 | 必须真实累计至少20个交易日；当前从1/20开始 |

更详细状态见 [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md)，每日操作见 [`docs/DAILY_SOP.md`](docs/DAILY_SOP.md)。

## 策略结构

### 第一层：AI_SCORE

| 分项 | 权重 | 计算逻辑 |
|---|---:|---|
| AI 龙头动量 | 40% | NVDA、AVGO、AMD、ANET、SMCI的20/60日动量滚动百分位 |
| SOX趋势 | 25% | 动量、MA60和MA120 |
| QQQ趋势 | 20% | 大型成长股风险偏好 |
| 利率环境 | 15% | 美国10年期收益率20日变化的反向百分位 |

| AI_SCORE | 市场状态 | 市场上限 | 单票最终上限 |
|---:|---|---:|---:|
| >=75 | 强风险偏好 | 100% | 80% |
| 60—74.9 | 正常 | 70% | 70% |
| 45—59.9 | 谨慎 | 30% | 30% |
| <45 | 防御 | 0% | 0% |

### 第二层：STOCK_SCORE

| 分项 | 最高分 | 主要依据 |
|---|---:|---|
| 趋势 | 30 | 收盘/MA20、MA20/MA60、MA60/MA120、收盘/MA120 |
| 动量 | 25 | 20日与60日收益率滚动百分位 |
| 量价 | 15 | 20日/60日成交量比及正收益确认 |
| 突破 | 15 | 收盘价相对120日最高价的位置 |
| 风险质量 | 15 | ATR波动百分位与60日回撤 |

只有`STOCK_SCORE >= 55`、收盘价高于MA60且MA20高于MA60时，才允许使用第一层给出的市场仓位。60日回撤达到15%减半，达到22%或跌破MA120清仓；持仓后还执行3倍ATR追踪止损。

## 数据准备

Choice日线Excel放在：

```text
data/input/choice_300308.xlsx
```

字段应包含证券代码、交易时间、开盘价、最高价、最低价、收盘价、成交量和成交额。原始Excel和标准化行情默认不上传GitHub。

## 第一次安装

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

也可以手动执行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH="src"
.venv\Scripts\python.exe check_environment.py
.venv\Scripts\python.exe -m pytest -q
```

## 运行方式

完整研究回测与稳健性分析：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe run_backtest.py
```

每日收盘后信号与模拟盘：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe run_daily.py
```

紧急停止模拟执行：在项目根目录创建空文件`STOP_TRADING`，或设置环境变量`KILL_SWITCH=true`。停机时仍可读取数据和报告，但不执行模拟订单。

## 主要输出

- `outputs/latest_signal.json` / `latest_signal.txt`：最新评分、风控和最终仓位
- `outputs/data_quality_report.json`：数据质量闸门
- `outputs/time_alignment_audit.csv`：未来函数日期审计
- `outputs/equity_curve.csv` / `trade_log.csv` / `metrics.csv`：严谨回测明细
- `outputs/equity_curve.png` / `drawdown.png`：净值和回撤图
- `outputs/backtest_metrics.json`：回测指标
- `outputs/benchmark_metrics.csv`：完整策略与三类基准比较
- `outputs/sample_split_metrics.csv`：训练、验证、样本外指标
- `outputs/cost_stress.csv`：1倍、1.5倍、2倍成本压力
- `outputs/parameter_stability.csv`：参数扰动结果
- `outputs/walk_forward.csv`：滚动走步结果
- `outputs/research_summary.json`：机械验收闸门
- `outputs/paper_daily_summary.json`：模拟盘当日摘要
- `outputs/next_order_plan.json`：下一交易日计划单
- `outputs/daily_reconciliation.csv`：每日目标与模拟持仓对账

大体量CSV、模拟账户状态和日志只保留在本机，不提交GitHub。

## 测试

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m pytest -q
```

目前共19项测试，覆盖评分边界、未来函数、Choice字段、下一日执行、一字涨停、ATR止损、费用、研究划分和模拟盘幂等性。