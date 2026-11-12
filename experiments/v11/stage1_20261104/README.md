# V11 SA-all 阶段 1 已完成材料

Hopper SA-all，逻辑种子 201–210，各 500k。201–206 复用，207 从零重跑，208–210 新训练；原中断 207 不计入正式结果。
Last5：SA-all 14.46743，O1 53.52034，T 42.53079，BC50 54.04944。按预设 H_act 判定不成立；SA-top 正式训练和阶段 2 未启动。

`stage1/` 是报告、配置、完成和独立核验凭据。`report_code/` 是 CPU 汇总修复脚本。
完整 STAGE1_RESULT.json 为 5,216,269 字节，未上传，保留在本机原位置；原始路径模板与 SHA-256 见 SNAPSHOT_SOURCES.json 的 excluded 项。
`tables/export_v11_tables.py` 直接从原 JSON 导出 70 条逐种子得分和组别统计。原 JSON 没有回合长度，不补造字段。
`tables/per_seed_scores_and_lengths.csv` 与 SA-all 子表从独立、已完成的 EPISODE_LENGTH_RESULT.json 关联 Last5 / Final 回合长度；两份来源及脚本 SHA-256 均记录在导出凭据中。该长度诊断只覆盖 SA-all、O1、T、raw1，其他组标为未记录。
所有导出均为只读 CPU 操作，新增训练和环境步为 0。模型、原始 TB/回合日志、逐步损失及向量不在此快照内。

原路径中的用户名已在公开副本中替换为 <USER>；源文件保持原样。原 FROZEN/完成凭据内引用的哈希仍是原始科学文件哈希，公开副本另有 snapshot_sha256。
