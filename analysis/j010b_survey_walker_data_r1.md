# j010b_survey_walker_data_r1 分析：失败（代码错误，训练前退出；已预期）

- 状态：failed，退出码 1。北京时间 2026-11-21 22:15:43 在 GPU 0 上启动，约 1 秒后退出；没有训练、没有回报数据。（relay 记录的结束时间 22:36 是下一轮轮询的时间。）
- 目标数据是齐的：`outputs/TARGET_DATA.json` 显示 8 个 walker2d 任务的目标文件都在 `~/odrl_target_extra_20261121/raw`，skipped 为空。
- 原因：与 `j010a_survey_hopper_data_r1` 完全相同——`controller.py` 的 `prepare()` 第 106 行遗留旧版本校验 ODRL 压缩包的一行，`ARCHIVE` 未定义 → `NameError`。上一会话（12:45 UTC）已预判此失败，见 analysis/j010a_survey_hopper_data_r1.md。
- 修复已在上一会话完成：`j010b_survey_walker_data_r2`（parent 为本任务）只删除这一行，任务、组别、种子 3126–3128 与 PLAN.md 的 j010 登记相同；它已于北京时间 22:36:24 在 GPU 0 上开始运行（上限 16 小时，预计约 9 小时，应在集群停机前结束）。
- j010 是描述性调查，不判定，不计入研究实验编号、“连续无筛查有效”和“连续失败”计数。本次不需要任何新操作。
