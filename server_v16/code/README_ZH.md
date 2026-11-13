# v16_shift：不同偏移类型下的源轨迹可跟踪性筛查

只用 CPU，不训练，不占 GPU，不需要任何目标数据集。hopper 20 个任务、walker2d 21 个任务（各含一个无偏移对照），
每个任务跟踪 500 条源轨迹的前 400 步。预计 30 分钟到 1 小时。

依赖服务器上已有的 `$HOME/v14_diag`、`$HOME/v13_action_fix/stage0` 和迁移项目里的两份 D4RL 源数据，全部只读。
环境 XML 来自 ODRL 仓库（MIT 许可），原样放在 `assets/`。

```bash
cd "$HOME"
sha256sum v16_shift.zip          # 与交付时给出的 SHA-256 核对
test ! -e "$HOME/v16_shift" && unzip v16_shift.zip
tmux new -A -s v16_shift
bash "$HOME/v16_shift/run_in_tmux.sh"
```

后台继续跑：Ctrl+B，松开，再按 D。看进度和结果：

```bash
d="$(cat "$HOME/v16_shift/outputs/LATEST_RUN.txt")"
cat "$d/STATUS.json"
cat "$d/REPORT_ZH.md"        # 完成后
cat "$d/FAILED.json"         # 出错时，原样发给我，不要重跑
```

规则在结果产生前写定，见 `config.json` 和报告开头。代码作者：Claude。本地用合成物理测试过，未在服务器上运行过。
