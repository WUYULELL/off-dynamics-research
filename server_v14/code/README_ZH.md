# V14 CPU 诊断包

本包仅运行 v14 的 A/B/C。V13 阶段 2、任何训练、其他任务均不启动。
由用户在内网手动上传、运行；本包不含 SSH 连接、账号密码、数据集、模型或环境安装程序。

## 已冻结的范围

- A：四组 BC-all / BC-fix / BC-mix / BC-mix-sel、601–605 最终模型。A1 全部目标 5000 行与同一固定 20000 条修正源样本；A2 读取末 5000 步损失并核对完整采样流；A3 各模型 10 回合，共 200 个新 CPU 诊断回合，保存全部观测与动作。
- B0：目标数据全部 8 个记录段，含末尾部分轨迹。执行跟踪动作后的实际下一观测与记录下一观测，逐步原始 L-infinity 误差必须 <= v13 config 的 corrected_observation_linf（2e-6），到达记录末尾且终止标签一致。任一不通过则 B1/B2 不启动，A/C 继续完成并汇报。
- B1/B2：仅当 B0 全部通过，自动运行。固定种子无放回选 500 条完整源轨迹；不足则用全部。原动作开环与当前实际状态闭环分别执行，均无模型训练。
- C：缓存全部 canonical 行按原轨迹内步数分档。
- 报告含预设解读、B0 无条件保留的逐轨迹结果、终止步数直方图、CPU/内存、保护核对和停点，不提出或启动下一项实验。

## 依赖与保护

只使用已经配置的 `$HOME/conda-envs/offdynamics38` 和现存 V13 文件。**无需 conda/pip 安装，无需登记 GPU。**
脚本设置 CUDA_VISIBLE_DEVICES=-1、MUJOCO_PY_FORCE_CPU=1，每进程 BLAS/PyTorch 单线程。
默认最多 16 个 B 工作进程；CLI 最多 24，加控制器及辅助进程低于 32。
使用前后核对原迁移包、V13 代码、25 项模型/日志及全量缓存、Gym/MuJoCo 原安装目录与 Conda 元数据的 SHA-256。
MuJoCo 绑定只在新输出 private_runtime 副本中导入/编译，不修改原环境。

代码异常、非有限数值、哈希不符、技术核对失败时停止，不自动重试。B0 的科学判据不满足仅阻止 B1/B2，按用户后续补充要求 A/C 照常运行。
不终止或操作任何他人进程；如本轮 B 子进程出错，只清理本控制器创建的子进程。

## 上传和运行

将外层 ZIP 通过 FinalShell 上传到自己的 `$HOME`。校验 ZIP 时使用外层运行说明给出的 SHA-256。
只向不存在的 `$HOME/v14_diag` 解压，不覆盖旧目录。解压后：

```bash
python3 "$HOME/v14_diag/verify_package.py"
```

未进入 tmux 时先执行：

```bash
tmux new -A -s v14_diag
```

进入后单独执行（已经位于 tmux 就跳过前面的进入命令，避免嵌套）：

```bash
bash "$HOME/v14_diag/run_in_tmux.sh"
```

入口自动激活原环境并设置全部路径，不需要重复手动 export。
顺序为只读保护核对 → B0 → C →（B0 通过时 B1/B2）→ A → 再次核对 → 汇总报告 → 退出。
三项允许并行，但本包采用上述顺序以降低同时内存占用，并避免加载 Torch 后再创建仿真进程。
刚启动的哈希核对和私有绑定准备期间可能暂时没有逐轨迹进度，勿重复启动。
用 Ctrl+B，松开，再按 D 可离开 tmux；重新查看用 `tmux attach -t v14_diag`。

## 查看进度与结果

每次重新登录均重新赋值 d，不能依赖前一个终端的变量：

```bash
d="$(cat "$HOME/v14_diag/outputs/LATEST_RUN.txt")"
if [ -f "$d/FAILED.json" ]; then
  cat "$d/FAILED.json"
elif [ -f "$d/COMPLETED.json" ]; then
  cat "$d/COMPLETED.json"
else
  cat "$d/STATUS.json"
fi
```

另一个终端也可只看 tmux 输出，不会重启任务：

```bash
tmux capture-pane -p -t v14_diag -S -15
```

完成后：

```bash
d="$(cat "$HOME/v14_diag/outputs/LATEST_RUN.txt")"
cat "$d/COMPLETED.json"
cat "$d/REPORT_ZH.md"
```

`completed=true` 表示本轮已到停点并生成汇报；还需看 `B0_passed` 与 `all_three_diagnostics_completed`。
B0 未通过时会标记 `B0_FAILED_A_C_COMPLETED_AWAITING_USER_REVIEW`，不将未运行的 B1/B2 说成完成。

通过 FinalShell 下载本次 `diagnostics_日期_时间` 目录即可交付所有表格、逐步记录与图。无需下载原 V13 数据或环境。
若只先发文字材料，请发 REPORT_ZH.md、COMPLETED.json、B/B0_RESULT.json、A/A2_log_check.json 与 INPUT_PROTECTION_AFTER.json 的 passed/文件数。

## 可追溯性与测试范围

FROZEN.json 固定本包；V13_CODE_EXPECTED.json 固定此前已部署源码。运行前写出完整 config.json（含样本位置与轨迹编号），看到结果后不更改判据。
随包 LOCAL_ACCEPTANCE.json 是本地合成 CPU 自检，不是 Linux/MuJoCo 真数据验收，更不是诊断结果。
本地检查包括阈值边界、空分箱、完整轨迹取样、实际状态累积、候选动作与最终执行分离、冻结策略 MSE。
正式 A/B/C 结果只有用户运行服务器入口后才会生成。
