# j010a_survey_hopper_data_r1 分析：失败（代码错误，训练前退出）

- 状态：failed，退出码 1。北京时间 2026-11-21 19:11:57 在 GPU 0 上启动，2 秒后退出；没有训练、没有回报数据。（relay 记录的结束时间 20:11 是下一轮轮询的时间。）
- 目标数据这次是齐的：`outputs/TARGET_DATA.json` 显示 8 个 hopper 任务的目标文件都在新目录里，SHA-256 与 `TARGET_SHA256.json` 一致，skipped 为空。
- 原因：`controller.py` 的 `prepare()` 第 106 行仍保留旧版本（j010a_survey_hopper）里校验 ODRL 压缩包的一行 `assert ARCHIVE.is_file() and sha(ARCHIVE) == ARCHIVE_SHA`，但新版本已删掉 `ARCHIVE` / `ARCHIVE_SHA` 两个常量 → `NameError: name 'ARCHIVE' is not defined`。本地合成测试走 `J010_LOCAL_TEST` 分支，跳过了这段代码，所以没测出来。
- `j010b_survey_walker_data_r1` 的 controller.py 与之相同（同一行），排到 GPU 后也会以同样的错误退出。
- 修复（本会话）：新建 `j010a_survey_hopper_data_r2`、`j010b_survey_walker_data_r2`（parent 为对应的 r1），只删除这一行，其余代码、任务、组别、种子（3126–3128，登记过但从未训练）都与 PLAN.md 的 j010 登记相同；目标数据仍逐个按哈希核验。推送前做了两项本地检查：(1) 用 AST 检查所有 .py 文件，没有其他未定义名称（r1 的检查结果正好报出 ARCHIVE / ARCHIVE_SHA）；(2) 用假工作进程完整跑通控制器流程（冒烟 → 主阶段 → RUNS.csv / summary.json / REPORT_ZH.md）。
- j010 是描述性调查，不判定，不计入研究实验编号、“连续无筛查有效”和“连续失败”计数（这是一次性的遗留代码错误，与之前的失败不同类）。
- 时间：每个 r2 预计约 9 小时（上限 16 小时）；排到卡后应能在集群停机（北京时间约 11-22 23:00 最后收一次结果）之前结束。若被停机打断，恢复后按规则重跑。
