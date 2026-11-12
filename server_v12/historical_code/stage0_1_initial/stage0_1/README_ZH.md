# V12 阶段 0.1 Linux 目标环境回放验收

本包只执行 V12 的 0.1，包含 Hopper-morph 和 Walker2d-morph。完成后硬停止，等待用户审阅结果并确认 0.2；不创建学习器、优化器或策略，不训练，不启动 0.3、阶段 1A/1B、ROOT 阶段 2/3 或任何旧队列。不修改 offdynamics38，不改原迁移包和旧结果。由用户在 FinalShell 手动上传和运行，助手不连接服务器。

## 本次范围及计数

两任务各有 5000 行目标数据；以固定种子分别有放回抽取 10000 次，重复样本不冒充独立数据。Hopper seed=2026110601，Walker seed=2026110602，写入 config_template.json；运行前复制为新输出的 config.json。没有根据回报选样或改设置。

目标观测的速度在正负 10 截断，Hopper 290 行、Walker 1890 行。直接把观测中的速度当完整 qvel 不可靠。因此按 AST 原样加载之前 Windows 目标回放验收使用的 step、recover 和状态转换函数：非截断观测恢复状态；截断行由同一轨迹内已验证的前驱回放恢复。不能跨 terminal 或 next_observations 与相邻 observation 不连续的边界。原轻量迁移包未包含这两个历史验收辅助脚本，本小包 references/source_snapshot 中另带原文件的逐字节副本并冻结哈希，不向原迁移包添加或修改文件；仅提取所需函数，不执行旧脚本顶层代码。

状态恢复预期另需 290+1890=2180 次环境步，两任务各 10000 次抽样回放另需 20000 次，Gym 0.18.3 两次构造各隐含 1 步，正常无失败时预计合计 22182 次环境步。异常时按实际已执行计数报告，不称 0 环境步。无新增真实数据优化或合成优化更新，GPU 测试 0，不导入 Torch。

本次只用目标数据原显式 next_observations。奖励和 terminal 的事实核对也保存，但不计算策略回报或新增 oracle 缓存。

## 验收判定和停点

Windows 原目标全 5000 行回放：Hopper 状态最大误差约 1.7461e-6，中位数约 2.5248e-7；Walker 最大误差约 3.9180e-5，中位数约 3.7678e-7。不能把所有行都强行要求小于 1e-6。

逐任务报告抽样误差 max/median/p99、全数据及抽样恢复失败行、实际独立行数、与 Windows 的最大值和中位数比，以及逐样本记录。历史技术门槛 state<=1e-3、reward<=1e-4、terminal 一致只作为旧验收核对；不是自行制定的 V12 新放行门槛。具体恢复失败或超过旧硬门槛时不进入下一个任务。

V12 的“误差明显变大”没有数值门槛。本入口无论结果如何都不会启动 0.2：成功完成采集也记 AWAITING_USER_REPLAY_REVIEW、v12_environment_acceptance_approved=false。用户先看报告，再决定是否通过及确认下一项。

## 为什么复制 mujoco_py 到新输出

mujoco-py 在导入时可能写 build-lock 或重新编译、修改扩展 rpath。为满足“不修改现有 Conda 环境”，本入口先只读盘点已安装的 mujoco_py，再完整复制到本次输出/private_runtime/mujoco_py，之后仅导入这个副本。编译或锁文件如发生均写入新副本，不向旧环境安装依赖，不改第三方源码。副本字节数、文件数与源文件哈希落盘；原安装目录运行后再次核对。原 MuJoCo 共享库只读使用。若依赖或编译失败，留证退出，不自动安装、重试或改环境。

全迁移项目 MANIFEST 的 1638 个文件及清单本身，在运行前后各核对一次。6 个 Gym 文件再次与原参考逐字节核对。CPU 线程固定 1；CUDA_VISIBLE_DEVICES=-1、MUJOCO_PY_FORCE_CPU=1。nvidia-smi 只保存只读资源快照，不创建 GPU 计算上下文。

## 上传和执行

这是新包，不要重新上传约 508 MB 的迁移数据。把 v12_linux_stage0_1_20261106.zip 上传到个人主目录。先按聊天给出的 SHA256 校验，再解压一次。若主目录已有 v12_linux/stage0_1 或指定输出，先查看并反馈，不覆盖、不删除。

在 FinalShell 中：

```bash
cd "$HOME"
sha256sum v12_linux_stage0_1_20261106.zip
python3 -m zipfile -t v12_linux_stage0_1_20261106.zip
test ! -e "$HOME/v12_linux/stage0_1"
```

最后一条无输出且返回成功，才解压：

```bash
python3 -m zipfile -e v12_linux_stage0_1_20261106.zip "$HOME"
tmux new -s v12_stage0_1
```

进入 tmux 后，一条命令完成激活、变量设置与 0.1：

```bash
bash "$HOME/v12_linux/stage0_1/run_in_tmux.sh"
```

脚本自身不会创建或修改已有环境。无需重复安装，不需要 chmod。正常结束或报错均停下，不再次启动。退出时按 Ctrl+B，松开，再按 D 可脱离 tmux；不要用 Ctrl+C 中断验收。

重新连接后查看：

```bash
tmux attach -t v12_stage0_1
cat "$HOME/v12_linux/outputs/stage0_1_20261106/REPORT_ZH.md"
cat "$HOME/v12_linux/outputs/stage0_1_20261106/STAGE0_1_COMPLETED.json"
```

如果失败，发送终端报错和 FAILED.json，保留整个目录。发生重启等中断也只做检查，不自动恢复或重跑。

## ROOT 代码定位

已只读定位文档指定仓库 https://github.com/cristianhtvc/ROOT 。README 标明基于 OTDF，依赖 jax==0.4.9、jaxlib==0.4.9、ott-jax==0.4.5、jaxopt==0.8.3，入口 run_root.sh。本次记录 main commit 0ae9c05d70ed369c16732c71abda59b54b559da9。

这只是代码定位，不表示已完成 ROOT 的数据/XML/指标核对。阶段 2 经用户确认后再在新目录固定实际使用的 commit、建立独立环境并逐项核对，不在当前 offdynamics38 安装 JAX。
