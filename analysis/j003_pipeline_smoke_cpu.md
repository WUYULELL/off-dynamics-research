# j003_pipeline_smoke_cpu 分析（2026-11-18 22:20 UTC，Claude 定时会话）

## 做了什么
只用 CPU，按 V21 的方式设置 MuJoCo 环境变量、使用 mujoco_py 私有副本，校验并解压 ODRL 目标数据，按 MANIFEST 校验源数据与学习器文件，对三个 V21 任务构建环境、做 1000 步 IQL 混合更新、500 步 BC 混合更新和一回合评估。冒烟种子 2999，回报不报告。

## 结果
- 状态 `completed`，退出码 0；脚本实际运行约 30 秒（02:11:38–02:12:08，北京时间），relay 于 06:11 取回。
- 8 项检查全部通过（`all_passed: true`）：FROZEN 哈希、MuJoCo 运行库、mujoco_py 私有副本导入（2.1.2.14，0.4 秒）、ODRL 压缩包（8 个条目）、MANIFEST（iql_dyn.py、iql_unc.py、hopper/walker 源数据哈希一致）、三个任务。
- 数据描述（与 V20/V21 一致）：hopper 腿关节 medium 目标 13 段、平均归一化回报 40.64（SD 16.70），源 2187 段、平均 44.40；walker 大腿关节 medium 目标 5 段 77.50、hard 目标 5 段 81.06，walker 源 1191 段、平均 61.97。V21 质量筛选保留行数 532,435（medium）、237,180（hard），与 V21 报告一致。
- CPU 速度：1000 步 IQL 混合更新约 7 秒，1000 步 BC 混合更新约 1.5 秒。自动任务内 `os.cpu_count()` = 128（j001 的 nproc=4 是当时进程可见核数，两者口径不同）。
- 附带发现：ODRL 压缩包中 hopper 腿关节 easy 与 hard 的目标文件哈希相同（V21 报告也指出这一点），这两个任务暂不使用。

## 按规则的结论
PLAN.md 登记：状态 `completed` 且 `all_passed` 为 true 即通过。**j003 通过**。后续训练任务沿用 j003 的环境设置（MuJoCo 环境变量、mujoco_py 私有副本、数据解压到任务自己的目录）。

## 局限
只在 CPU 上验证；GPU 路径由 j004 自带的冒烟阶段（5 个组各 1000 步、一个评估点）在正式训练前检查。
