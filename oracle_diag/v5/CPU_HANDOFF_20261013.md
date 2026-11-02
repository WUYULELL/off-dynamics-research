# Oracle v5 当前状态

CPU 准备完成。没有 v5 GPU 授权文件，没有启动 v5 GPU 训练或队列。
用户要求完成并交付实验，但提供的 v5 文件明确要求 GPU 前询问，以及冒烟汇报后确认正式训练。
下一步须获得用户对 A、B GPU 冒烟、复用验证和并行吞吐测试的确认。

## 文件与执行

- 协议：`../results/v5_20261013/protocol.json`，不得改科学设置、seed、预算或判据。
- 清单：`../results/v5_20261013/training_plan.json`，30 项正式，8 项单独冒烟，2 项并行冒烟。
- A 入口：`train_entry_v5.py`；Python 为 `D:/ANACONDA/envs/offdynamics/python.exe`。
- B 入口：`reproductions/otdf_v5/train_entry_v5.py`；Python 为 `reproductions/otdf_train/Scripts/python.exe`。
- 队列：`run_queue_v5.py --stack A|B --kind smoke|formal --phase standalone|concurrent|formal --launch <新的启动日志目录>`。
- 每队列有操作系统文件锁防重复启动；每项输出须为空。失败不得覆盖，从相同 seed 的 0 步在新 attempt 路径重跑。
- 始终用 `-B` 禁止旧目录生成 pycache。后台启动用隐藏窗口。只检查本实验实际命令行，不终止其他应用。

## 已完成证据

- A CPU：`cpu_a1/completed.json`；真实 source+target，路径不变性、梯度排除、v4/v5 10 步精确对齐。
- B CPU：B 结果目录 `cpu_b3/completed.json`；两种正则开关逐步核验、零 source 抽样、官方 OTDF-NOREG 抽样和过滤、历史文件哈希。
- A5：`a5_2/completed.json`；全部 5000 个 target 状态、15 个已完成 actor、精确 k=5；12 个固定点独立穷举核对。
- `existing_baselines.json`：15 个 A 历史基线、5 个 B 历史基线从原始 TensorBoard 重读。
- `PREPARATION_v5.md` 与对应 `preparation_report_evidence.json`，不属于完成正式实验的报告。
- 新代码可在 GPU 冒烟之前因发现的工程错误修复，保留修复前后哈希和失败日志；需要重发 cpu_ready.json 时保留旧版本到新 attempt，不能变更科学协议。

## 确认后顺序

1. 记录本任务用户确切授权消息，写 `gpu_smoke_authorization.json`：authorized=true、scope=smoke、protocol_sha256、授权原文。不要自行伪造授权。
2. 先单独跑 A standalone 队列，再单独跑 B standalone 队列；它们的第一个任务都是历史复用检查。不通过立即停下报告。
3. 两条 concurrent 冒烟队列同时跑，每个进程 2 CPU 线程；不会覆写独立冒烟。
4. `audit_smoke_v5.py` 重读所有日志、参数和模型，核对复用与吞吐，产生 smoke_audited.json。用实际吞吐汇报 ETA；B 若 >72h，只提出后两组可缩为 seeds 1–3 的方案，不自动改清单。当前默认仍为完整五种子。
5. GPU 正式须另有用户确认，记录 `gpu_formal_authorization.json`（scope=formal）。然后两条 formal 队列并行。
6. 中期：两边首组各 5 seed 完成后，`report_v5.py --stage interim`，人工检查 QA1/QA2/QB1 并交付，留下交付凭据；继续队列。
7. 吞吐低于单独的一半，脚本写 defer_B_until_A10.json 并在 B 作业边界等待 A 前十项；当前 B 活跃训练保留。若用户要求立即暂停，需另行安全处理当前进程状态，不删除或拼接训练结果。
8. 全部完成：`report_v5.py --stage final` 独立重算后，人工逐项检查报告、模型与数据证据，补足 QA/QB 综合解释与描述性比较，再交付 `REPORT_v5.md`。如建立过跟进自动任务，交付后删除。

## 解释限制

A/B 不横向排名：A 500k/5000 条 explicit_next，B 1M/4999 条官方 next-row。
A 只切断直接损失，source 仍可通过学得的 Q、V 间接影响后续 actor。
QA2 不证明唯一/主要通道，QA3 不证明理论上限，未过门槛不证明等效或无用。
B 保留官方 actor 广播行为；QB2 还包含有效 batch 128→230 的改变；VAE 预训练还改变后续 RNG 序列。
A5 的动作距离只提供描述性证据，不等同于回报或因果效应。

CPU 准备故障证据：a5_1（旧 SciPy workers 不支持，精确单线程修复）、B cpu_b1（同名模块冲突，改为 otdf_methods_v5）。旧训练未动。
