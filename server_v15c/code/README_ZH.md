# v15c_walker 使用说明

walker2d 形态偏移的源轨迹跟踪（B1、B2），只在前 H = 400 步内判定。只用 CPU，不训练，预计一两分钟。
依赖服务器上已有的 `$HOME/v15_screen`（只读）和 V15 的输出 `screen_20261113_102214`。

```bash
cd "$HOME"
sha256sum v15c_walker.zip          # 与交付时给出的 SHA-256 核对
test ! -e "$HOME/v15c_walker" && unzip v15c_walker.zip
tmux new -A -s v15c_walker
bash "$HOME/v15c_walker/run_in_tmux.sh"
```

后台继续跑：Ctrl+B，松开，再按 D。看结果：

```bash
d="$(cat "$HOME/v15c_walker/outputs/LATEST_RUN.txt")"
cat "$d/STATUS.json"; ls "$d"
cat "$d/REPORT_ZH.md"        # 正常完成时
cat "$d/FAILED.json"         # 出错时，原样发给我，不要重跑
```

规则（看到任何 walker 源轨迹结果之前写定）见 `config.json` 和报告开头。代码作者：Claude。本地用合成物理测试过，未在服务器上运行过。
