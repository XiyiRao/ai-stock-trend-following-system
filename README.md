# AI Stock Trend Following System

这是一个每日收盘后运行的市场状态识别系统。当前阶段按照《中际旭创_量化自动交易系统_完整实施操作指南》实现了数据质量闸门、严格的中美交易日对齐，以及四因子 AI 市场状态评分。

> 本项目仅用于研究和教学，不构成投资建议。真实交易前必须接入可靠行情，并独立验证复权、时区、停牌、滑点、税费和涨跌停规则。

## 已完成范围

- 美股 AI 篮子：NVDA、AVGO、AMD、ANET、SMCI
- 市场指标：SOX、QQQ、美国 10 年期国债收益率
- A 股标的：中际旭创（300308），前复权日线 OHLCV/成交额
- 中国交易日历与数据质量检查
- 严格跨市场对齐：A 股日期 t 只能使用满足 US_Date < CN_Date(t) 的最近美股数据
- 10 个历史日期的确定性时间对齐审计
- 下一交易日执行的研究型基准回测

## AI_SCORE

| 分项 | 权重 | 计算逻辑 |
|---|---:|---|
| AI 龙头动量 | 40% | 20 日、60 日平均收益率的 3 年滚动百分位，60%/40% 合成 |
| SOX 趋势 | 25% | 20 日动量百分位、价格高于 MA60、MA60 高于 MA120 |
| QQQ 趋势 | 20% | 20 日动量百分位、价格高于 MA60、MA60 高于 MA120 |
| 利率环境 | 15% | 10Y 收益率 20 日变化的反向滚动百分位 |

滚动百分位窗口为 756 个交易日，至少需要 252 个有效观察值，只使用当时及以前的历史数据。

| AI_SCORE | 市场状态 | 市场仓位上限 | 中际旭创最终上限 |
|---:|---|---:|---:|
| >= 75 | 强风险偏好 | 100% | 80% |
| 60–74.9 | 正常 | 70% | 70% |
| 45–59.9 | 谨慎 | 30% | 30% |
| < 45 | 防御 | 0% | 0% |

## Choice Excel 数据

如果无法使用 Choice API，可将中际旭创日线导出为 Excel，并保存为：

    data/input/choice_300308.xlsx

导出字段需要包含：证券代码、交易时间、开盘价、最高价、最低价、收盘价、成交量、成交额。系统会自动忽略空行和 Choice 页脚，验证证券代码与 OHLCV，并优先使用该文件；文件不存在时才回退到 AKShare/本地缓存。原始 Excel 默认被 Git 忽略，不会上传 GitHub。

## 安装与运行

需要 Python 3.10 或更高版本。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
python run.py
```

## 主要输出

- `outputs/latest_signal.json`：最新市场状态和目标仓位
- `outputs/data_quality_report.json`：字段、日期、有效成分、文件哈希与质量闸门
- `outputs/time_alignment_audit.csv`：10 个日期的未来函数审计
- `outputs/us_market_scores.csv`：美股原始评分时间序列
- `outputs/market_state.csv`：对齐到中国交易日的状态序列
- `outputs/backtest.csv`：下一交易日执行的研究型基准回测
- `outputs/market_dashboard.png`：价格、AI_SCORE 与仓位上限图

大体量行情 CSV 默认只保存在本地，不提交 GitHub。仓库保留最新信号、质量报告和仪表盘，方便直接查看。

## 测试

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
```

目前的回测只是验证信号时序和仓位映射的基准，不等同于可实盘使用的完整回测。下一阶段应实现个股多因子 `STOCK_SCORE`、严格撮合、停牌与涨跌停处理、滑点税费模型，以及样本外验证。
