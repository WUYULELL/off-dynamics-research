# v20_check：两项低成本检验（方案 C）

1. **源数据与目标数据的质量对比**（只用 CPU，几分钟）：在 3 个关节限位任务上统计 D4RL medium 源轨迹和 ODRL 目标轨迹的归一化回报、长度、每步奖励，看能否解释 V18b 中“hopper 上混合有益、walker 上混合有害”。
2. **预训练-微调动力学模型的确认**（GPU，约 2 小时）：V19 中观察到“源数据预训练、目标数据微调的模型（FT）比只用目标数据的模型（T）更准”，但那不是预设判定条件。本次用新种子 2001–2005、**整条轨迹留出**重新检验。只比较模型误差，不训练策略、不运行仿真器。

判定只看 hopper-morph、walker2d-morph：两个任务上 FT 都优于 T（配对差 > 0 且 > 2 倍配对标准误）为“确认”，否则“不确认”。规则见 `INSTRUCTION_V20.md` 和 `config.json` 的 `rule`。

## 先决条件

`odrl_target_20261115.zip` 仍在主目录。

## 运行

```bash
cd "$HOME"
sha256sum v20_check.zip
test ! -e "$HOME/v20_check" && unzip v20_check.zip
tmux new -A -s v20_check
bash "$HOME/v20_check/run_in_tmux.sh"
```

后台继续：Ctrl+B，松开，再按 D。不要按 Ctrl+C，也不要在这个窗口里再输入启动命令。

## 查看

```bash
d="$(cat "$HOME/v20_check/outputs/LATEST_RUN.txt")"
python3 -c "import json;s=json.load(open('$d/STATUS.json'));print(s['time'],s['phase'],{k:s[k] for k in ('pending','running','done','failed') if k in s})"
cat "$d/REPORT_ZH.md"     # 完成后
cat "$d/FAILED.json"      # 出错时，原样发给我，不要重跑
```

代码作者：Claude。
