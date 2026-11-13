# v15b_b0diag：V15 B0 自检的只读诊断

不仿真、不训练、不占 GPU，几秒钟完成，不需要 tmux。只读取 `$HOME/v15_screen/outputs/screen_20261113_102214` 和目标数据，
输出写在 `$HOME/v15b_b0diag/outputs/` 下。

```bash
cd "$HOME"
sha256sum v15b_b0diag.zip          # 与交付时给出的 SHA-256 核对
test ! -e "$HOME/v15b_b0diag" && unzip v15b_b0diag.zip
bash "$HOME/v15b_b0diag/run.sh"
d="$(cat "$HOME/v15b_b0diag/outputs/LATEST_RUN.txt")"
cat "$d/REPORT_ZH.md"
```

出错时把终端里的报错原样发给我，不要重跑。

用户于 2026-11-12 授权（选项 A）。只描述，不作判定；walker 的限时规则在看到本诊断结果后、看到任何 walker 源轨迹结果之前写定。
代码作者：Claude。本地用合成数据测试过，未在服务器上运行过。
