# 自动循环日志

每个任务结束后追加一条：日期｜任务｜假设 → 结果 → 结论 → 下一步。

---

## 2026-11-18 15:21 UTC（Claude 定时会话）
- 新结果提交 61920fa 只是状态变化：j001_smoke_cpu、j002_collect_v21 进入 running，j000_smoke 仍 waiting_gpu。没有已结束的任务，无需分析。
- j001 尚未 completed，按第 4 步前提不提交研究实验。下一步：等 j001 结果。

## 2026-11-18 17:20 UTC（Claude 定时会话）
- j001_smoke_cpu：假设“CPU 链路通、资源路径存在” → completed，numpy/torch/gym 可导入，三个资源路径都在 → 按规则通过（见 analysis/j001_smoke_cpu.md）。附带发现：自动任务环境没有设置 MuJoCo 路径，mujoco_py 导入失败（V21 用 `$HOME/.mujoco210/mujoco210` + 私有副本）；4 张卡利用率 96–99%。
- j002_collect_v21 仍在运行（等待 V21 结束）；j000_smoke 仍 waiting_gpu。
- 下一步：提交 j003_pipeline_smoke_cpu（CPU，上限 1 小时），验证 MuJoCo 设置、数据、学习器和三个 V21 任务的环境。暂不提交研究训练任务，理由：训练链路在自动任务环境里尚未验证，而且 GPU 当前全满；V21 结果出来前可做的“不依赖 V21 的实验”也需要先有可用的链路。j003 通过后再设计第一个研究实验。

## 2026-11-18 18:21 UTC（Claude 定时会话）
- 新结果提交 ce5d680 只是状态变化：j003_pipeline_smoke_cpu 于 02:11（北京时间）进入 running（上限 1 小时）；j002 仍在运行；j000 仍 waiting_gpu。没有新结束的任务，无需分析。
- 下一步不变：等 j003 结果，通过后再设计第一个研究实验。

## 2026-11-18 22:40 UTC（Claude 定时会话）
- 新结果提交 5fd7417：j002_collect_v21、j003_pipeline_smoke_cpu 完成；j000_smoke 在 3 号卡上运行。
- j003：假设“自动任务环境能跑通完整训练链路” → 8 项检查全部通过 → 通过（analysis/j003_pipeline_smoke_cpu.md）。
- V21（经 j002 取回，按 V21 自己的规则，数字由 RUNS.csv 独立重算）：
  - 第一部分：hopper 腿关节 medium 上 BC-混合**确认**（54.12 对 BC 41.19，+12.93，SE 1.64，10/10）；混合-全部**未确认**（+6.30，SE 3.24）。
  - 第二部分 H_Q：walker hard 上按规则“支持”（BCMIXQ +14.03，5/5），但未筛选的 BC-混合同样有效且更高（74.10 对 71.79），筛选本身无增益；walker medium 上不支持（BCMIXQ 71.41 低于 BC 73.71，BC-混合 58.83 不稳定）。
  - 结论：BC-混合是两个任务上的最强对照；按回报筛选不是可用的方法方向。
- 下一步：提交 j004_domain_cond_bc_screen（DCBC、QDCBC 两个候选，与 T、BC、BC-混合同机同种子比较；种子 3001–3005；1 卡，上限 22 小时）。规则见 PLAN.md 登记。
