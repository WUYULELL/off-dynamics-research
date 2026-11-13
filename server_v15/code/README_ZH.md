# v15_screen 使用说明

只用 CPU，不训练，不占 GPU。预计几分钟。

## 部署

把 `v15_screen.zip` 上传到服务器的个人主目录，然后：

```bash
cd "$HOME"
sha256sum v15_screen.zip          # 与交付时给出的 SHA-256 核对
test ! -e "$HOME/v15_screen" && unzip v15_screen.zip
```

如果 `test` 那一行没有任何输出、unzip 也没执行，说明目录已存在，先停下告诉我。

## 运行

```bash
tmux new -A -s v15_screen
bash "$HOME/v15_screen/run_in_tmux.sh"
```

后台继续跑：按 Ctrl+B，松开，再按 D。不要按 Ctrl+C。

## 查看结果

```bash
d="$(cat "$HOME/v15_screen/outputs/LATEST_RUN.txt")"
cat "$d/STATUS.json"; ls "$d"
cat "$d/REPORT_ZH.md"        # 正常完成时
cat "$d/FAILED.json"         # 出错时
```

- phase 为 `STOPPED_AWAITING_USER_REVIEW` 且有 `COMPLETED.json`：已完成，把 `REPORT_ZH.md` 发给我。
- 有 `FAILED.json`：出错停下了，把内容原样发给我，不要重跑。

## 它会做什么

1. 核对本包、`v14_diag`、`v13_action_fix/stage0` 和所用数据文件的哈希。
2. hopper-morph：B0 自检，再重跑 v14b 的 B1/B2 作回归检验。
3. 回归一致后，walker2d-morph：B0 自检，通过后跑 B1/B2。
4. 再次核对原始文件未被改动，生成报告并停下。

所有输出都在 `$HOME/v15_screen/outputs/` 下，不修改其他任何目录。
