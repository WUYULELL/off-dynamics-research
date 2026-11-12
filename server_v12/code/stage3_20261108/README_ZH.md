# V12 阶段 3 服务器运行说明

本包使用服务器现有的迁移数据、阶段 1A、1B、2 结果和两个 Conda 环境。只写入新的 `~/v12_linux/outputs/stage3_20261108`。GPU 为此前已登记使用的物理 0 号 A6000，按 UUID 识别。按服务器规则登记自己的条目，再在 tmux 中运行。

```bash
tmux new -s v12_stage3
bash "$HOME/v12_linux/stage3_20261108/run_in_tmux.sh" --registered
```

启动时先保护性核对旧文件、写入三并发预算并输出 CPU 真实批次核验。OT 求解器与训练种子的独立性通过后，单独计算一次权重。权重子进程退出后才运行训练，任何一批最多三个训练进程，每个进程的 PyTorch 和 BLAS 线程数为 1。指定卡存在其他任务时每 60 秒重新检查并记录等待。若准入信息无法确认或进程故障，保留现场并停止放行。

运行期间可在另一个终端查看以下文件。文件随进度生成。

```bash
d="$HOME/v12_linux/outputs/stage3_20261108"
cat "$d/PRELAUNCH_REPORT_ZH.md"
cat "$d/ROOT_Q_READONLY_ZH.md"
cat "$d/ROOT_WEIGHT_COMPARISON.json"
tail -n 10 "$d/GPU_WAIT.jsonl"
cat "$d/STAGE3_COMPLETED.json"
cat "$d/STAGE3_REPORT_ZH.md"
```

`STAGE3_COMPLETED.json` 中 `completed: true` 表示 15 项审计与汇总通过。`FAILED.json` 记录队列失败，已有结果保留。`GPU_WAIT.jsonl` 仅在卡被占用后生成。Ctrl+B 再按 D 可离开 tmux，会话保留。阶段 3 正常退出后检查自己的进程已经结束，再移除自己的登记条目。
