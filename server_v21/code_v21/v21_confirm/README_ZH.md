# v21_confirm：hopper 有效组的确认 + walker 按回报筛选源数据的筛查

- **第一部分（确认）**：V18b 中 hopper 腿关节 medium 上“混合-全部”和“BC-混合”按规则有效，但只是 5 个种子的筛查。本次用 10 个新种子（2101–2110）重测 T、BC、混合-全部、BC-混合。
- **第二部分（筛查，新假设 H_Q）**：walker 两个大腿关节任务上，V18b 混合无效，V20 发现源轨迹平均回报低于目标轨迹。本次只保留“记录回报不低于目标轨迹平均回报”的源轨迹，再做混合（混合-全部-筛选、BC-混合-筛选），与 T、BC 和未筛选的混合组比较。5 个新种子（2201–2205）。筛选只用离线数据，不用仿真器。

共 100 项训练（IQL 50 项、BC 50 项），GPU 规则与 V18b 相同。预计约 15–25 小时，视显卡占用而定。

## 先决条件

`odrl_target_20261115.zip` 仍在主目录。

## 运行

```bash
cd "$HOME"
sha256sum v21_confirm.zip
test ! -e "$HOME/v21_confirm" && unzip v21_confirm.zip
tmux new -A -s v21_confirm
bash "$HOME/v21_confirm/run_in_tmux.sh"
```

后台继续：Ctrl+B，松开，再按 D。不要按 Ctrl+C，也不要在这个窗口里再输入启动命令。

## 查看

```bash
d="$(cat "$HOME/v21_confirm/outputs/LATEST_RUN.txt")"
python3 -c "import json;s=json.load(open('$d/STATUS.json'));print(s['time'],s['phase'],{k:s[k] for k in ('pending','running','done','failed') if k in s})"
cat "$d/REPORT_ZH.md"     # 完成后
cat "$d/FAILED.json"      # 出错时，原样发给我，不要重跑
```

规则见 `INSTRUCTION_V21.md` 和 `config.json` 的 `rule`。代码作者：Claude。
