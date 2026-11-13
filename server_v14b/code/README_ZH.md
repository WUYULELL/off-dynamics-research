# v14b_track 使用说明

只用 CPU，不训练，不占 GPU。预计几分钟到十几分钟。

## 部署

把 `v14b_track.zip` 上传到服务器的个人主目录，然后：

```bash
cd "$HOME"
sha256sum v14b_track.zip          # 与交付时给出的 SHA-256 核对
test ! -e "$HOME/v14b_track" && unzip v14b_track.zip
```

`test ! -e` 是为了保证解压到一个不存在的新目录；如果它没有输出而 unzip 也没执行，说明目录已存在，先停下告诉我。

## 运行

```bash
tmux new -A -s v14b_track
bash "$HOME/v14b_track/run_in_tmux.sh"
```

想让它在后台继续跑：按 Ctrl+B，松开，再按 D。不要按 Ctrl+C。

## 查看结果

```bash
d="$(cat "$HOME/v14b_track/outputs/LATEST_RUN.txt")"
cat "$d/STATUS.json"
ls "$d"
cat "$d/REPORT_ZH.md"        # 正常完成时
cat "$d/FAILED.json"         # 出错时
```

- `STATUS.json` 的 phase 为 `STOPPED_AWAITING_USER_REVIEW` 且存在 `COMPLETED.json`：已完成，把 `REPORT_ZH.md` 发给我。
- 存在 `FAILED.json`：出错停下了，把它的内容原样发给我，不要重跑。

## 它会做什么

1. 核对本包和 `$HOME/v14_diag` 的文件哈希。
2. 读取 v14 已保存的 B0 逐步数组，按修订后的门槛重新判定（不仿真）。
3. 通过后，对 500 条源轨迹做 B1 开环重放和 B2 闭环跟踪。
4. 前后核对 v13 模型、缓存、代码和环境没有被改动。
5. 生成报告并停下。

它不会修改 `v14_diag`、`v13_action_fix`、迁移项目或 Conda 环境里的任何文件；所有输出都在 `$HOME/v14b_track/outputs/` 下。
