# V12 阶段 0.2 Hopper T 学习器迁移验收

用户于 2026年11月6日确认“0.1 通过，继续 0.2”。本包只覆盖该项，完成后停在 AWAITING_STAGE0_3_CONFIRMATION，不运行 0.3 或阶段 1/2/3，也不启动 SA-top、旧队列或正式实验。由用户在 FinalShell 上传并运行；助手不连接服务器。

## 固定工作量

1. determinism_a：Hopper T，seed201，10000 学习器更新，无环境或评估。
2. determinism_b：同一配置从零独立运行10000更新，另一个新进程，无环境或评估。
3. 仅前两项逐步损失与最终状态确定性通过后，entry_20k 从零运行20000更新；10k和20k各评估10回合。

三项串行，共40000学习器更新、120000次Adam更新；3个学习器、9个Adam对象、3个目标采样器。CPU审计只加载保存状态，不创建或更新模型。仅第3项创建1个Hopper目标环境；20个评估回合最多20000步，另记Gym构造器隐含1步，按实际计数报告。没有额外GPU探针、预热优化、参考训练或自动重试。完成状态会区分“短跑验收”和“正式实验”。

## 科学实现

从原迁移包 source_snapshot/oracle_diag/coverage_raw_r14 按AST原样提取seed、model_config、target_data、Learner、Sampler、initialize、evaluate；原IQL_DYN和iql_unc模块按MANIFEST哈希核对后加载。不执行旧Windows runtime的顶层代码或旧pipeline，不修改原文件。

生成的学习器配置必须逐项等于迁移包中R14 Hopper T seed201的learner配置：11/3维、两层256隐藏单元、温度10、expectile .7、gamma .99、tau .005、actor/critic lr3e-4、target batch64。源数据不参与V/Q/actor，不加载源HDF5或oracle。target显式next_observations，不重缩放观测或reward。每次更新都更新target_Q；actor为pre-V优势加权动作MSE，不是-logpi。

短跑仅提前终止循环，max_step/cosine仍为500000，不缩成10000或20000。目标评估为原OTDF XML、timestep .002、frame_skip4、TimeLimit1000；每次10回合，seed500000到500009，保存并恢复Python/NumPy/Torch CPU/CUDA训练RNG，额外核对sampler未变。原evaluate已经记录逐回合长度，新增摘要仅在结果detach之后计算。

预先固定：torch.use_deterministic_algorithms(True)、CUBLAS_WORKSPACE_CONFIG=:4096:8、cudnn benchmark=False、cudnn deterministic=True。matmul与cudnn的allow_tf32固定True，保持Torch1.11的已记录默认设置；所有设置进入config，失败后不自动切换并重跑。CPU各线程库及torch线程/interop均为1。

## 比较与审计

两次10k逐步V/Q/actor按float32位比较，并核对JSONL字节、初始化、完整target流、最终四网络参数和三Adam/cosine完整状态。checkpoint文件序列化SHA只用于文件完整性，不将其当参数数值等价的唯一证据。若损失不同，报告首个不同的步和字段；若仅最终参数不同，报告最终检查点差异，不能推断未保存的最早参数分岔步。不同即停，不自动运行20k，不自动改确定性配置或重试。

第三项前10000步另与第一项比较损失、参数及采样，检查完整入口和环境创建没有改变训练前缀。每项自动CPU审计所有模型/Adam/scheduler有限性、每个Adam步数、cosine周期与lr、原始JSONL和losses.npy一致、完整采样重建，以及评估长度/归一化/RNG。

迁移包未保留该Hopper T seed201的原始逐步损失或TB。Windows部分写“无可对照记录”，不从audit摘要构造损失轨迹，也不为此上传本机旧日志或额外训练。跨Windows/Linux旧结果只作描述，不作同种子配对推断。

报告逐项披露损失min/median/p99/max和最后值，没有结果前数值门槛的“量级合理”不事后编造阈值。20k只有两个评估点，Last5不适用；短跑分数与长度只用于检查入口。

## 资源与旧文件保护

传入的GPU必须是管理员或资源组分配给你的卡。控制器解析成UUID；每个GPU子进程显式设置CUDA_VISIBLE_DEVICES为同一UUID，逻辑设备仅cuda:0。所有GPU任务串行；控制器与CPU审计设置CUDA_VISIBLE_DEVICES=-1。启动前和每项前保存设备快照，所选卡显存非0或利用率非0时拒绝启动，不切换到别人的卡、不终止其他用户进程。空闲查询不能替代预约。

原迁移包1638文件及MANIFEST在前后核对。0.1报告及全部凭据核验后，在新目录写用户通过授权，旧0.1文件不修改。原Conda的包元数据、mujoco_py目录、MuJoCo共享库和已验收6个Gym文件在前后核对。20k评估如需mujoco-py导入锁或编译，复制绑定到新的entry_20k/private_runtime中使用；不修改原环境，不安装依赖。所有科学代码、配置和输入哈希记录。中断或新故障留证停止，不自动重启/恢复。

## 上传运行

仅上传新的 v12_linux_stage0_2_20261106.zip 到个人主目录，不重传原数据包、不重复解压0.1。

校验并只解压到新的 v12_linux/stage0_2。若该目录或stage0_2_20261106输出已存在，先查看，不覆盖。打开tmux：

```bash
tmux new -s v12_stage0_2
```

确认仍分配给自己的是GPU0且当前空闲后，运行：

```bash
bash "$HOME/v12_linux/stage0_2/run_in_tmux.sh" 0
```

最后的0是物理GPU编号，不是阶段编号；实际分配不同就传入分配的编号或完整UUID。脚本会自动激活原研究环境并设置路径。不要重复pip/conda安装。Ctrl+B再D脱离tmux后可断开FinalShell；任务仍在服务器运行。

查看终端与最终结果：

```bash
tmux attach -t v12_stage0_2
cat "$HOME/v12_linux/outputs/stage0_2_20261106/REPORT_ZH.md"
cat "$HOME/v12_linux/outputs/stage0_2_20261106/STAGE0_2_COMPLETED.json"
```

正常凭据completed=true、jobs_completed=3、learner_updates=40000、policy_evaluation_episodes=20、stage0_3_started=false。结束后把报告和凭据发回聊天。出错发送FAILED.json和相关stderr日志，保留现场，不自动重跑。tmux会话仍存在不代表训练仍在运行。

本机准备阶段仅编译、AST、固定流重建和纯数组/状态字典测试；0新优化更新、0真实环境步、0GPU计算。服务器运行耗时尚未实测，以此次3项记录为准，不把旧Windows吞吐当服务器实测。
