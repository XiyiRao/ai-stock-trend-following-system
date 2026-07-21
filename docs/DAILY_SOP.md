# 模拟盘每日SOP

## 每日收盘后

1. 在Choice导出中际旭创前复权日线，覆盖`data/input/choice_300308.xlsx`。
2. 打开PowerShell并进入项目目录。
3. 执行：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe run_daily.py
```

4. 查看以下文件：

- `outputs/data_quality_report.json`：必须`passed=true`。
- `outputs/latest_signal.txt`：阅读AI分、个股分、风控和最终仓位。
- `outputs/next_order_plan.json`：确认下一交易日模拟计划。
- `outputs/paper_daily_summary.json`：确认模拟账户和累计天数。
- `outputs/daily_reconciliation.csv`：目标股数与实际模拟股数差异。

## 系统如何推进模拟账户

- T日收盘后生成T+1计划单。
- 下一次出现新的A股交易日数据时，系统用新交易日开盘价执行上一张计划单。
- 同一个信号日期重复运行不会重复计数或重复下单。
- 单日最多按配置的单笔金额和换手率推进目标仓位，大仓位可能分多天完成。
- 一字涨停不能模拟买入，一字跌停不能假设卖出，停牌不成交。

## 紧急停止

在项目根目录执行：

```powershell
New-Item STOP_TRADING -ItemType File
```

恢复前先核对账户和日志，然后删除：

```powershell
Remove-Item STOP_TRADING
```

也可以临时设置：

```powershell
$env:KILL_SWITCH="true"
```

停机开关存在时，系统仍生成数据和报告，但计划状态为`disabled`，不执行模拟订单。

## 异常处理

- 数据质量失败：不手工修改结果，检查Choice字段和日期。
- Yahoo限流：系统可使用本地缓存，但必须确认缓存日期。
- 目标与模拟持仓不一致：先查看`paper_trades.csv`和`daily.log`，禁止直接伪造成交。
- 重复运行提示锁存在：确认没有另一个程序运行，再检查`work/paper_daily.lock`。
- 连续异常：启用STOP_TRADING，保留日志，停止推进模拟盘。

## 20日验收

`paper_daily_summary.json`中的`twenty_day_validation_complete`只有累计20个不同交易日后才会变为`true`。达到20天后仍需人工确认期间没有未解释异常，才能认为第12章运行验收完成。