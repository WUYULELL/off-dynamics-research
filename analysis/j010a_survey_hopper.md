# j010a_survey_hopper 分析：失败（没有可用的目标数据）

- 状态：failed，退出码 1（北京时间 2026-11-21 14:31:52 启动，1 秒后退出；GPU 3）。没有进行任何训练，没有回报数据。
- 原因：控制器 `prepare()` 检查 8 个登记任务的目标文件 `<任务>_medium.hdf5`，ODRL 压缩包里一个都没有（`outputs/TARGET_DATA.json`：kept 为空，8 个任务全部 “target file not in the ODRL archive”），触发 `assert kept, 'no task has target data'`。
  - 缺的任务：hopper-gravity-0.5 / 2.0、hopper-friction-0.5 / 2.0、hopper-kinematic-footjnt-medium / hard、hopper-morph-foot-medium / hard。
- 这是登记时没有先核对压缩包内容造成的（主研究对话 2026-11-21 06:50 已说明），不是代码缺陷；代码按登记规则“缺文件就去掉、不替换”执行，行为正确。
- j010_survey 是描述性调查，不判定，不计入研究实验编号和“连续无筛查有效”计数。按 CONTEXT.md 第 6 节“任何子任务失败则整组不判定”，j010 整组作废。
- j010b_survey_walker 已由主研究对话把 `max_hours` 改为 25，停在 `waiting_approval`，**不要授权**（多半同样缺数据）。
- 下一步（由主研究对话 / 用户推进，不由定时会话做）：请 Codex 列出 ODRL 压缩包里全部 hopper / walker2d 目标文件，再按实际存在的数据用新任务名重新登记调查（不复用 j010 这个名字）。种子 3126–3128 虽未实际用于训练，但已登记，按规则视为已用，新登记从 3129 起。
