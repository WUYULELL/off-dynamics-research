# v19_residual：残差动力学模型试点

问题：在稠密偏移（形态改变）下，用 100 万条源数据学到的动力学，加上只用 5000 条目标数据学到的“两个环境之差”，能否比只用目标数据训练的模型更准确地预测目标环境？

- 只训练动力学模型，比较它们在留出目标数据上的预测误差。**不训练策略，不运行仿真器**。
- 四种模型：S（源模型直接用）、T（只用目标数据）、FT（源模型在目标数据上微调）、RES（冻结源模型 + 残差网络）。
- 任务：hopper-morph、walker2d-morph（稠密偏移，参与判定）；3 个 ODRL 关节限位任务（稀疏偏移，只作参照）。
- 5 次重复（种子 1901–1905），每次重复重新随机划分目标数据；训练比例 25%、50%、100%。
- **会使用 GPU**，每个进程显存不到 2 GB。预计 30 到 60 分钟。

## 先决条件

`odrl_target_20261115.zip` 仍在主目录（试点 A 用过的同一个文件，不用解压，程序会核对哈希）。

## 运行

```bash
cd "$HOME"
sha256sum v19_residual.zip
test ! -e "$HOME/v19_residual" && unzip v19_residual.zip
tmux new -A -s v19_residual
bash "$HOME/v19_residual/run_in_tmux.sh"
```

后台继续跑：Ctrl+B，松开，再按 D。不要按 Ctrl+C。

## 查看

```bash
d="$(cat "$HOME/v19_residual/outputs/LATEST_RUN.txt")"
python3 -c "import json;s=json.load(open('$d/STATUS.json'));print(s['time'],s['phase'],{k:s[k] for k in ('pending','running','done','failed') if k in s})"
cat "$d/REPORT_ZH.md"        # 完成后
cat "$d/FAILED.json"         # 出错时，原样发给我，不要重跑
```

冒烟测试（几分钟）失败时，请同时把 `$d/runs/` 下出错任务目录里的 `stdout.log` 发给我。
规则在结果产生前写定，见 `INSTRUCTION_V19.md` 和 `config.json` 的 `rule`。代码作者：Claude。
