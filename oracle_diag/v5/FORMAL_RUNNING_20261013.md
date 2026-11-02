# Oracle v5 正式训练运行交接

更新：2026-10-13T17:54:28.109889+08:00。用户明确授权：“启动 A、B 全部 30 项正式训练，完成后继续更新桌面的跨域离线强化学习研究文档”。完整 A/B 30 项已授权并启动，不再等待正式确认；旧 CPU_HANDOFF 和冒烟报告的待确认描述是历史状态。

## 实际执行

- 读取结果目录 active_execution.json 和 A/B queue_status.json，再核对实际进程命令行；PID 可能重用，不能只凭 PID 判断。
- 两个队列均为 run_queue_v5.py --stack A|B --kind formal --phase formal；队列内部逐组逐 seed 执行，A/B 同时运行。
- 2026-10-13 17:42（北京时间）启动。初始队列 PID：A=46752，B=17664；B 虚拟环境可能有 Python 启动器和实际训练子进程，不能误判重复队列。
- 计划：oracle_diag/results/v5_20261013/training_plan.json；冻结协议 protocol.json；正式授权 gpu_formal_authorization.json。
- A 日志 formal_launch1/A；B 日志 formal_launch1/B（均在 A 结果目录）；队列 stderr/stdout 在 formal_launch1 根目录。
- A 输出 oracle_diag/results/v5_20261013/f1；B 输出 reproductions/otdf_v5/results_20261013/f1。
- 启动前复核10项冒烟、全部3037项旧文件、冻结18脚本与冒烟交付输入哈希均通过；证据 formal_launch1/prelaunch_verified.json、startup_verified.json。
- 初次启动检查：A CRITIC-ORACLE seed0 已到 31000 步，B IQL-TARGET seed1 已到 48000 步。这是快照，以最新 progress 为准。

## 冻结范围和后续交付

A：CRITIC-ORACLE → CRITIC-SOURCE → Q-ORACLE，每组 seeds0–4，各500k，64+64抽样，source系数1.0。B：IQL-TARGET → IQL-TARGET+REG → OTDF-NOREG，每组seeds1–5，各1M；启用REG另VAE预训练10k。保持完整30项，不自动缩种子，不更改损失、预算或判据。不得改旧代码和结果，不得重跑冒烟或历史基线。

每项GPU结束后由 audit_v5.py 在CPU审计；审计失败先修审计，不应重跑已完成训练。真实训练故障保留失败目录，从相同seed第0步在新attempt恢复，不拼接、不按回报挑选。新代码修改须记录前后哈希和依据。

progress_v5.py 生成 interim_ready/final_ready；首组A/B各五seed全部审计完成后，运行 report_v5.py --stage interim，人工独立核对QA1/QA2/QB1并交付中期报告，记录交付凭据，队列继续。正式吞吐低于单独一半时产生 defer_B_until_A10.json，在B作业边界等待A前10项，保留当前B训练并通知用户。

最终运行 report_v5.py --stage final，并独立核对原始TensorBoard：A460k–500k、B960k–1M Last-5；逐seed差、样本SD、SE、正号数和Final。补充两项描述性比较、分组均值及对QA/QB的清晰解释。核验实际shifted评估、归一化、抽样、初始化、模型有限性、source/oracle/OT行对齐及3037保护哈希。脚本拒绝覆盖已存在报告，重审时须保留旧版本并写新attempt证据。

2SE且至少4个正号为工程门槛，不能声称严格95%置信结论、数学上界、等效或所有source有害。A/B分数不可横排，A5仅描述性；QA2只检验移除直接source actor损失，source仍可通过Q/V间接影响actor；QB2包含batch变化，VAE会改变后续RNG。保留单任务、target约8条轨迹段、V为学得函数、oracle恢复隐藏状态等解释范围。

## 桌面资料库维护

目标：C:/Users/TXin/Desktop/跨域离线强化学习研究。本轮只更新入口、研究思路、当前边界与启动状态，并备份更新前页面到99_历史归档/旧版文档/20261013_正式训练启动前。研究进度总表.xlsx和已复制的冒烟/CPU报告当前仍是归档时快照。

完成后继续更新00总览中的研究进度总表.xlsx、研究问题与当前思路.md、文件位置索引.md，按03实验记录、04结果分析、05汇报材料放置新报告与轻量证据，更新清单和哈希记录。不要搬动原始数据模型，不要覆盖用户新增修改。保留报告旧版和更新前文档。表格更新应用spreadsheets技能；参考 outputs/research_archive_20261013 下构建/验证脚本，但不要直接重跑拒绝已有根目录的 build_archive.py。当前归档核验记录只证明最初归档快照，之后更新另有增量核验凭据。

自动跟进ID oracle-v5，每小时检查；正常推进保持安静，中期、完成、故障或需要用户操作时通知。交付最终报告与桌面资料链接并完成更新后删除自动跟进。
