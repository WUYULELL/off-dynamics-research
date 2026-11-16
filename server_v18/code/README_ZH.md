# v18_pilotA：试点 A

在 ODRL 的三个关节限位任务上，比较只用目标数据（T、BC）和混合 D4RL medium 源数据（混合-价值、混合-全部、BC-混合）。
每组 5 个种子（801–805），共 75 项训练；开始前先跑 10 项各 2000 步的冒烟测试。**会使用 GPU**，预计 5 到 6 小时。

## 先决条件

`odrl_target_20261115.zip` 已上传到主目录（不用解压，程序会核对哈希并解压到 `$HOME/v18_pilotA/data/`）。

## 运行

```bash
cd "$HOME"
sha256sum v18_pilotA.zip odrl_target_20261115.zip
test ! -e "$HOME/v18_pilotA" && unzip v18_pilotA.zip
tmux new -A -s v18_pilotA
bash "$HOME/v18_pilotA/run_in_tmux.sh"
```

后台继续跑：Ctrl+B，松开，再按 D。不要按 Ctrl+C。

## 查看

```bash
d="$(cat "$HOME/v18_pilotA/outputs/LATEST_RUN.txt")"
python3 -c "import json;s=json.load(open('$d/STATUS.json'));print(s['time'],s['phase'],{k:s[k] for k in ('pending','running','done','failed') if k in s});[print(' ',k,v) for k,v in sorted(s.get('progress',{}).items())]"
cat "$d/REPORT_ZH.md"        # 完成后
cat "$d/FAILED.json"         # 出错时，原样发给我，不要重跑
```

冒烟测试失败会在几分钟内停下；失败时请同时把出错任务目录下的 `stdout.log` 发给我。
规则在结果产生前写定，见 `INSTRUCTION_V18.md`。代码作者：Claude。
