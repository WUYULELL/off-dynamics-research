# OTDF 本机复现准备状态

> 以下为启动前的历史准备快照。当前进展见 [RUN_STATUS.md](RUN_STATUS.md) 及 `runs/pipeline_5seeds_20260921/status.json`。seed 1 已成功完成；用户授权补齐其余种子，9 月 21 日 17:18 已启动 seed 2–5 串行队列。原有等待指令已被启动授权替代。

准备日期：2026-09-19。**已完成环境与脚本准备，等待用户明确通知启动。** 当前电脑在跑其他实验；没有运行数据导出、OT 求解、CVAE 训练、策略训练、环境评估或性能测试，也没有设置自动启动。

## 已准备

- 官方源码与作者提供的目标域数据：相邻目录 `../OTDF`。来源 <https://github.com/dmksjfl/OTDF>，锁定提交 `41e2e626844bd6b65e2316a418ce4d6df4bbe775`，官方算法文件未改动。
- 首任务：`walker2d-morph`，正常 D4RL medium 源域 → 形态变化 medium 目标域。源缓存原始 1,000,000 行，官方目标数据原始 5,000 行；训练加载后的有效行数等后续 Export 阶段记录。
- 训练隔离层：`../otdf_train`，Python 3.8.13；只读继承现有 offdynamics 的 PyTorch 1.11.0+cu113、MuJoCo 等包，Gym 0.23.1 单独安装在该隔离层。原环境仍是 Gym 0.18.3，没有更新其包。
- 独立 OT 环境：`../otdf_ot`，Python 3.12.14、CPU JAX/JAXlib 0.4.20、OTT 0.4.5、JAXopt 0.8.3。OT 函数直接取自已锁定官方源码，计算尚未执行。
- 启动器 `run_windows.ps1` 默认只显示状态。四个计算阶段 Export / Cost / Smoke / Train 均须以后显式启动，各阶段不会自动串联或后台运行。

## 检查结果与实际运行边界

已通过两个环境的 `pip check`、OT 依赖导入、官方训练模块及 Windows 路径包装器导入、Python/PowerShell 静态语法检查。导入检查隐藏 CUDA，未构造策略模型、未创建环境实例、未做反向传播或求解。

**“环境检查通过”不等于“论文已复现”。** 后续还需要验证 MuJoCo 实例创建、CPU OT 的 float16 数值与耗时、训练执行和评估结果。Smoke 为 1000 步策略更新，但仍含原版 10000 步 CVAE 预训练及初始环境评估；它不是当前可偷偷运行的轻量检查，也不能作为论文效果。

## 硬件判断

本机为 Ryzen 5 5600（6 核/12 线程）、约 16GB 系统内存、RTX 2060 SUPER 8GB。适合在已有实验结束后，先串行验证一个任务。准备时可用内存快照约 2.93GiB，已有训练在运行，因此当前不并行增加实验。

8GB 显卡能否满足实际峰值、CPU OT 需要多久，仍待启动后的测量；这里没有编造耗时估计。正式单种子使用 100 万步策略更新；论文效果核对还需 5 个随机种子及相同协议的基线，不能用单次或短跑结果下结论。

## 复现口径

作者 README 指定 PyTorch 1.8、JAX/JAXlib 0.4.9。本机为 Windows 兼容方案，PyTorch 和 JAX 版本不同，OT 用 CPU。因此应标注为**保留官方算法的 Windows 兼容复现**，不能声称与作者软件环境完全相同。

论文描述与官方源码还存在 Actor 优势加权形状、硬筛选影响范围等差异；首轮保留官方实现，不静默修正。详见 [协议审查](PROTOCOL_AUDIT.md)。此前讲解中“筛选仅作用于 Q”是按论文公式描述；官方代码的硬筛选还会影响 V 和 Actor 收到的样本，软权重直接进入 Q 损失。

## 文件入口

- [启动与阶段说明](README_SETUP.md)
- [启动器](run_windows.ps1)
- [协议与代码差异](PROTOCOL_AUDIT.md)
- `prepared_status.json`：机器、数据、依赖与验证快照。
- `training_packages.json`、`ot_packages.json`：当前有效包版本和位置。
- `requirements_train_overlay.txt`、`requirements_ot_windows.txt`：专用依赖约束。

等待用户通知后，先检查已有实验是否结束和可用资源，再按阶段启动。不建立提醒、排队训练或自动重试。
