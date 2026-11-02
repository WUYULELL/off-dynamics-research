# Windows 启动与数据桥接脚本

Walker2d-morph seed 1 已完成，Final=36.92，Last-5=46.86。用户已授权补齐 seed 2–5，串行队列于 2026-09-21 17:18 启动。当前进展见 [RUN_STATUS.md](RUN_STATUS.md)、`runs/pipeline_5seeds_20260921/status.json`，逐种子结果见该目录的 SUMMARY.md/summary.json/per_seed.csv。旧中断过程及全部模型保留。

## 默认安全入口

当前五种子流程由 `run_remaining_seeds.ps1` 管理，已运行，不要重复启动。它先核验固定 `study_five_seeds.json`、现有 seed 1 和数据，随后顺序执行 seed 2–5 的完整训练、检查和聚合。`-ValidateOnly` 只做预检；本次预检已通过。`aggregate_seeds.py --study <清单> --output <目录> [--validate-seed N]` 只读取结果并生成汇总，不启动模型。

统计口径见 [FIVE_SEED_PROTOCOL.md](FIVE_SEED_PROTOCOL.md)：Final 为主、Last-5 为辅助，分别跨 5 个独立种子计算均值和样本标准差 ddof=1。正在运行或尚未结束的种子不纳入汇总；只有实际完成证据齐全才算完成。

在 PowerShell 中调用 `& '.\reproductions\otdf_setup\run_windows.ps1'`，默认 `-Stage Status` 只查看文件是否存在、打印后续命令；不启动 Python、环境 rollout、OT 或训练。

`run_windows.ps1` 可分别运行以下阶段（已完成或正在运行的阶段不要重复启动）：

```powershell
& '.\reproductions\otdf_setup\run_windows.ps1' -Stage Export
& '.\reproductions\otdf_setup\run_windows.ps1' -Stage Cost
& '.\reproductions\otdf_setup\run_windows.ps1' -Stage Smoke -Seed 1
& '.\reproductions\otdf_setup\run_windows.ps1' -Stage Train -Seed 1
```

路径应从项目根目录执行；也可以使用启动器的绝对路径。默认 D4RL 缓存为 `C:/Users/TXin/.d4rl/datasets`，MuJoCo 为 `C:/Users/TXin/.mujoco210/mujoco210`，可使用 `-DatasetDir`、`-MujocoDir` 覆盖。运行中只设置当前进程环境并在结束后恢复，清除继承的 `ODRL_*`、`PYTHONPATH/PYTHONHOME`；不会修改全局设置或其他正在运行的实验。

Smoke/Train 通过 `run_official.py` 进入原始 `train_otdf.py`：在 Python 进程内再次规范 MuJoCo/D4RL/PATH、移除父 ODRL 项目的同名包路径，再用 `runpy.run_path(..., run_name='__main__')` 运行原入口。这样规避宿主传递 PATH 时丢失 MuJoCo 路径的问题；没有更改算法或 monkeypatch。启动记录同时保存 wrapper 参数和原始训练器参数。

## 四个显式阶段

1. **Export**：使用 `otdf_train` 的 Python 3.8。调用官方 `d4rl.qlearning_dataset` 与 `algo.utils.call_tar_dataset`，按原顺序拼接 `[observations, actions, next_observations]` 保存为两份 `.npy`。构造 Gym 环境，但不做 rollout 或学习；源缓存必须已存在，不自动下载。清单记录处理后行数、维度、dtype、原数据/加载代码/特征文件 SHA256；不修改官方目标数据处理方式。
2. **Cost**：使用独立 `otdf_ot` 的 Python 3.12/JAX CPU 环境。AST 提取并执行官方 `make_otdf_cost.py` 中的 `solve_ot`，避免导入整份脚本的 Torch/Gym。保留 float16、cosine、max_cost、Sinkhorn threshold=1e-9/max_iterations=100、源分块 10000、每块完整目标数据及负成本符号。仅修复整除时的空尾块。文件写入官方训练器要求的 `OTDF/costlogs/walker2d-morph-srcdatatype-medium-tardatatype-medium.hdf5`，键为 `cost`；拒绝覆盖已有文件。保留文件旁边的 `.manifest.json`，训练前必须通过数据、代码、成本哈希核验。float16 若出现非有限值就停止，不自行改 dtype 或参数。
3. **Smoke**：使用原训练器，策略更新 10000 步，加原版 10000 步 CVAE 预训练、初始与第 10000 步环境评估。独立保存日志和模型，核验数值有限、目标域评估和模型保存成功；不能作为论文结果。
4. **Train**：单个 seed，默认 1，原版 1000000 策略更新步，保留 `--weight --proportion 0.8 --reg_weight 0.5 --save-model`。默认正式输出在 `otdf_setup/runs/train_1M`，Smoke 单列 `smoke_NOT_PAPER_RESULT`。启动前记录参数；已有 seed 目录或启动记录即拒绝覆盖，没有断点恢复保证。

## 本次授权的连续运行

`run_pipeline.ps1` 在 Export 完成后顺序执行 Cost → Smoke → 检查 → Train → 检查。每阶段独立日志，状态每 5 秒写入 `runs/pipeline_20260919_seed1/status.json`；任何阶段失败或数值检查不通过就停止，不自动重试。固定输出目录存在即拒绝重复启动，不会覆盖旧结果。它是本次单任务流程，不是定时或循环任务。

`inspect_run.py --run-dir <r1目录> --console-log <日志>` 可只读检查当前 TensorBoard/控制台记录，输出 JSON，包括实际观测步数、数值有限性、源/目标回报、归一化目标分数与速度估计。只有观察到进程确实结束后才可传入 `--exit-code`，不能将仍在运行的任务标成完成。

9 月 21 日恢复使用 `run_pipeline.ps1 -ContinueAfterSmoke`，它要求旧流程的 Cost/Smoke 已成功退出、只在 InspectSmoke 失败且未尝试 Train；复用原数据，重新检查原 Smoke 后从头启动正式训练，新记录写入 `pipeline_20260921_seed1`。`-ContinueAfterSmoke -ValidateSmokeOnly` 仅检查，在独立 `pipeline_20260921_validation` 记录验证结果，不启动训练。两者均拒绝覆盖已有目录；本次已执行，不要重复启动。

重启后新的尝试使用 `run_pipeline.ps1 -ContinueAfterSmoke -AttemptName restart_20260921_1052_seed1`，流水线和训练目录带独立名称，保留旧运行。内部仅向 Train 传 `-TrainRunName train_1M_restart_20260921_1052_seed1` 选择新输出目录；算法、超参数、seed 和完整训练步数不变。该命令已经执行，不要重复启动。此功能从头重跑，不是断点加载。

Windows 包装器在本次 Python 进程中设 `MUJOCO_GL=disable`，避免 D4RL 可选 dm_control 厨房环境在导入时初始化不用的 GLFW 渲染。物理模拟和官方算法未改。Python 以无缓冲模式运行，以便实时记录控制台日志。

## 复现边界

- 官方仓库提交 `41e2e626844bd6b65e2316a418ce4d6df4bbe775`；算法和环境代码未改。桥接脚本存放在相邻 `otdf_setup`。
- 原作者 README 的 JAX/jaxlib 为 0.4.9；本机 Windows CPU 使用 0.4.20，OTT 仍为 0.4.5。虽然 OT 公式及参数照原样保留，数值一致性尚未验证，须报告为 Windows 兼容复现。
- 训练环境 Python 3.8 只读继承现有 offdynamics 包，加独立 Gym 0.23.1 overlay；已有实验环境中的包不在这里修改。
- 首任务是正常动力学源域 `walker2d-medium-v2` → 形态变化目标域 `walker2d-morph`，两域 medium。一个 seed 跑通不能代表五个 seed 的论文结果复现。
- 论文协议、官方实现已知差异、原始回报归一化与最终评价口径见 `PROTOCOL_AUDIT.md`。本桥接没有修复那些算法/数据处理差异，也不改变官方训练日程。
- 中断后若留下 `.partial`、某 seed 的日志或 invocation 文件，先检查失败原因，不能直接覆盖或声称可恢复。输入数据/代码变化后须重新生成对应成本。
