# v17_reg：试点 B（小数据正则化）

在 hopper-morph、walker2d-morph 上，只用 5000 条目标数据训练 IQL 和 BC，比较三种做法：不加正则（base）、权重衰减（wd）、稀疏掩码（sparse）。
每种 5 个种子（701–705），共 60 项训练；开始前先跑 6 项各 2000 步的冒烟测试。

**会使用 GPU。** 只向"没有别人进程、且空闲"的卡放行任务，每张卡最多同时 6 项，四张卡最多 24 项。预计 3 到 4 小时。

```bash
cd "$HOME"
sha256sum v17_reg.zip          # 与交付时给出的 SHA-256 核对
test ! -e "$HOME/v17_reg" && unzip v17_reg.zip
tmux new -A -s v17_reg
bash "$HOME/v17_reg/run_in_tmux.sh"
```

后台继续跑：Ctrl+B，松开，再按 D。不要按 Ctrl+C。

```bash
d="$(cat "$HOME/v17_reg/outputs/LATEST_RUN.txt")"
python3 -c "import json;s=json.load(open('$d/STATUS.json'));print(s['phase'],{k:s[k] for k in ('pending','running','done','failed') if k in s});[print(' ',k,v) for k,v in sorted(s.get('progress',{}).items())]"
cat "$d/REPORT_ZH.md"        # 完成后
cat "$d/FAILED.json"         # 出错时，原样发给我，不要重跑
```

- 冒烟测试失败会立刻停下（一两分钟内）。训练代码没有在本地运行过，冒烟测试就是第一次真实运行。
- 被外部信号杀掉的任务会从第 0 步自动重跑（同种子、同配置、新目录，最多 3 次）；程序自身报错则停止放行新任务并报告。
- 对迁移项目、`v13_action_fix`、Conda 环境只读；所有输出在 `$HOME/v17_reg/outputs/` 下。

规则在结果产生前写定，见 `config.json` 和 `INSTRUCTION_V17.md`。代码作者：Claude。
