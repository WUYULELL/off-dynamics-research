# v5 GPU 冒烟已获授权并启动

用户在本任务确认：启动 A、B 的 GPU 冒烟、复用验证和并行吞吐测试。
授权凭据：oracle_diag/results/v5_20261013/gpu_smoke_authorization.json。

- 2026-10-13 启动 pipeline PID 26828；仅用于初始定位，后续务必核对实际命令行。
- pipeline：oracle_diag/v5/run_smoke_pipeline_v5.py。
- 日志：oracle_diag/results/v5_20261013/smoke_launch1。
- 顺序：A 独立四项→B 独立四项→A/B 并行两项→CPU 总审计。
- 首项 A ORACLE-W1 的 GPU 训练和实际 (64,64) 抽样已确认。
- 监测：本任务 heartbeat oracle-v5，每 10 分钟检查；正常无可行动变化不通知。
- 正式 30 项训练仍须在冒烟汇报后获得确认。不能由当前 smoke 授权推断正式授权。
- 所有修改与恢复仍限定在 v5；保留已完成和失败产物，复用不一致先停下报告。

状态以 active_execution.json、smoke_pipeline_status.json 与 A/B queue_status.json 指向的实际进程和日志为准。
