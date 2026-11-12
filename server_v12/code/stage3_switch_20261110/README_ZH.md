# V12 阶段 3 换卡续跑

此入口核验并复用 0 号卡上已经完成的训练和 OT 权重。剩余任务从获准的 1、2、3 号 RTX A6000 中选择一张，同一次接续及其后续重试都固定用该卡。T_s504 的中断目录作为证据保留，未完成的种子以原种子从第 0 步启动。逐种子的物理卡号与 UUID 写入正式报告和结果 JSON。

## 先退出仍在等卡的旧队列

在 FinalShell 普通终端接回旧会话：

```bash
tmux attach -t v12_stage3_resume
```

如果末尾仍在输出 `resume_selection: waiting`，在该会话内按 Ctrl+C，等 Shell 提示符出现。若已出现 `GPU started` 或 `wave_... active`，保留正在执行的队列。按 Ctrl+B 再按 D 回到普通终端，用以下命令检查：

```bash
ps -u "$(id -un)" -o pid,ppid,stat,args | grep -E '[s]tage3_(controller|worker)[.]py'
```

## 选择并登记新卡

```bash
bash "$HOME/v12_linux/stage3_switch_20261110/run_in_tmux.sh" --scan
```

扫描只读。按集群规则登记空闲卡，更新自己的原 0 号卡登记。单卡最多 3 个进程，共享卡只新增 1 个并保留原 60% 准入门槛。若准许从多张已获准的卡中自动选一张，可以将它们一并传给续跑命令。

## 启动

确认旧进程已退出后，从普通终端进入新会话：

```bash
tmux new -A -s v12_stage3_switch
```

在新会话内执行，`1` 换成实际登记卡号：

```bash
bash "$HOME/v12_linux/stage3_switch_20261110/resume_in_tmux.sh" 1
```

预期计划为 `8 fully trained, 7 remaining`，实际以入口核验结果为准。旧锁仍被占用时本入口退出。运行结果位于 `~/v12_linux/outputs/stage3_switch_20261110`，重复接续使用后缀 `_attempt02` 等新目录。`RESTART_PLAN.json` 记录来源，`STAGE3_COMPLETED.json` 表明 15 项全部审计完成。
