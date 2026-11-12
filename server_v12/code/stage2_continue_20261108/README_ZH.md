# V12 阶段 2 接续入口

本包依赖服务器已有的 `v12_linux/stage1_2_20261107` release 2 和已安装的 `conda-envs/v12_root_20261107`。接续前核验原阶段 2 为启动权重试算前的 GPU 准入失败，同时核验已完成的 1A、1B 和旧环境清单。CPU 重新比对配置与数据后，按原协议执行 ROOT 权重试算、VAE 预训练、20,000 步主循环和 CPU 审计。

按服务器的项目、实验、比赛优先级登记自己的条目，确认原分配卡可用后，在 tmux 运行 `bash "$HOME/v12_linux/stage2_continue_20261108/run_in_tmux.sh" --registered`。`--registered` 表示你已完成手动登记。入口要求同一张卡在各 GPU 子进程启动前通过原空闲检查，权重计算和训练各一个 GPU 进程，串行执行。原站点每卡 3 进程上限适用。

新报告和完成凭据位于 `$HOME/v12_linux/outputs/stage2_root_20261108_continue1/`。控制器等待自己启动的 GPU 进程退出并完成 CPU 审计后，写出 `COMPLETED.json` 并停在 `AWAITING_STAGE3_CONFIRMATION`。确认自己的程序退出后，删除自己的登记条目。若中途失败，保留新旧目录，查看 `FAILED.json` 和对应日志。若入口第一道 GPU 准入失败，终端会输出占用情况，新结果目录尚未创建。
