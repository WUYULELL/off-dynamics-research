# V12 阶段 0.3 单卡吞吐测量

用户于2026年11月7日授权“启动0.3”。仅准备并执行本节，不进入阶段1A/1B、ROOT环境或训练，不启动旧队列或正式实验。助手不连接服务器，由用户在FinalShell上传并手动运行。

## 工作量和固定设置

Hopper T IQL按并发1、2、4、6递增；每档种子从901起，分别901、901–902、901–904、901–906，固定配置和数据。每个进程20000更新，训练结束后只评估一次10回合。BC-all单进程seed901也20000 actor更新和一次10回合评估；“同样测速”解释为同样20k短跑，不运行完整200k。最大13次IQL加1次BC，260000 IQL更新、20000 BC更新、800000 Adam更新；最多140评估回合，构造器步另计。

IQL使用0.2已验收的原R14学习器、更新顺序、target batch64、温度10、cosine500000，source不参与。BC使用原V9阶段3 Actor/Sampler：原Policy、Adam3e-4、batch64、beta0、动作维均值再批均值MSE，不构造或训练Q/V，cosine200000不缩短。全部目标5000行含尾段，原raw观测/reward、OTDF XML、.002 timestep、TimeLimit1000、原归一化。短跑不充正式结果；Last5不适用，不按回报挑并发数。

科学函数和类按原快照AST原样加载；新代码只处理Linux路径、固定短循环、并发、计时、资源监测与报告。原旧Windows模块顶层和队列不执行。原target_data在两算法上都等于直接读显式next_observations。BC原ids函数读取已冻结的V9 selections.json，核对BC_all恰好为0–4999全部行。原seed函数保留：IQL用R14前缀，BC用V9前缀；评估也保留各自原函数的seed公式。

## 计时与并发

所有GPU进程显式使用同一已分配UUID，最大6个；每进程OMP/MKL/OpenBLAS/NumExpr、torch线程和interop为1。deterministic_algorithms、CuBLAS工作区及TF32设置与0.2相同。控制器和审计进程CUDA_VISIBLE_DEVICES=-1。

先让全部进程初始化模型、数据和目标环境并等待READY屏障，再同时放行训练。记录前10000/后10000步及训练总耗时。较快进程训练结束后等待全组训练完成，才放行一次评估，避免评估与其他进程训练混算。初始化、等待、评估、CPU审计单列，不放进纯训练耗时。没有额外优化预热或GPU探针。屏障原子发布且拒绝覆盖，避免读到未写完JSON；控制器消失时，worker检查父进程并自行停止。

报告两种吞吐：用户定义的各进程steps/s之和，以及整组总更新数除以同步放行到最后训练结束的时间。建议采用资源压力可接受档位中的整组吞吐最高者，未来正式并发上限仍需用户确认。

## 资源保护

沿用0.2通过的同一GPU UUID和驱动资源画像：空闲时0–3 MiB、利用率0、本卡计算及pmon进程均为空，连续3次核对。其他GPU进程不阻止本卡。每个新档位开始前重新准入，档位内部不把自己的训练进程误当外来占用。

读取nproc、affinity和可见cgroup层级的CPU配额；查询nproc时临时清除OMP变量对查询进程的影响，训练的线程限制不变。CPU并发不得超过min(6, floor(有效可见CPU核数)-1)，不足2核时最多单进程；cgroup限额无法解析时只测单进程。为共享CPU保留至少一核的预算（单核情形无可保留，明确记录）。这些是资源上限，不代表空闲核都已分配给本账号。

开始每档前要求可用内存至少2 GiB余量加2倍预测worker RSS；初次按每worker 1 GiB预算，之后使用实测峰值。GPU预计用量不超过80%总显存。每2秒记录GPU利用率/显存、仅本任务worker的CPU/RSS、host及affinity CPU忙碌比例、可见cgroup限流。host或affinity CPU忙碌>85%、被限流period比例>25%、内存余量<2 GiB持续15秒会阻止后续档位；本档正常完成。内存降至512 MiB紧急线或本任务出错时停止自己的worker并留证，不信号其他用户进程。

任何IQL档位被资源上限跳过时不越级测更高IQL档位；纯配额限制下可继续符合单进程条件的BC。持续资源压力或用户STOP_AFTER_TIER标记会跳过余下测速。未测档位明确列出，不外推其吞吐或将其当通过；all_planned_tiers_measured=false时只能给已测范围建议。

可在另一个终端手动要求“当前档完成后停止”：

```bash
touch "$HOME/v12_linux/outputs/stage0_3_20261107/STOP_AFTER_TIER"
```

## 输出和保护

新代码$HOME/v12_linux/stage0_3，新输出$HOME/v12_linux/outputs/stage0_3_20261107。旧包、Conda、旧0.2源码和输出不改。启动前解析0.2完整完成凭据，逐文件核验其evidence_sha256，确认3项审计、确定性和完整入口前缀均通过；旧JSON按完整语义核对，源码仍检查原字节SHA。结束再核对旧输入、原Conda/MuJoCo/Gym及0.2全部输出/源码。

各worker在新输出中使用mujoco_py副本，导入锁或编译不写原环境。每项结束自动CPU audit：模型/Adam/cosine有限、20000计数、完整采样、原始损失、一次评估长度/归一化和训练RNG保持。测试结果与正式目录分开；不重复0.1或0.2，不自动重试、恢复或重训。

输出REPORT_ZH.md、THROUGHPUT_RESULT.json、STAGE0_3_COMPLETED.json。正常全测状态AWAITING_USER_STAGE1_DECISION；资源受限AWAITING_USER_DECISION_RESOURCE_LIMITED并标all_planned_tiers_measured=false；新故障FAILED.json。只有完成凭据才表示本次控制器已结束；REPORT也会在失败时生成，不能单凭报告存在判断成功。

耗时外推：IQL500k=25倍20k训练时间+50次10回合评估；BC200k=10倍训练时间+40次评估。早期策略回合短，后期评估开销可能更高，这只是规划外推，不是精确运行时保证。根据0.2单进程IQL约100秒/10000步，全部IQL更新的串行等价约43分钟；实际并行与评估总耗时待本次实测。

## 上传与启动

只上传v12_linux_stage0_3_20261107.zip到个人主目录，按聊天给的SHA256校验解压到新目录，不覆盖旧目录。已在tmux里可直接运行；否则先tmux new -s v12_stage0_3。确认0号卡仍分配给自己后：

```bash
bash "$HOME/v12_linux/stage0_3/run_in_tmux.sh" 0
```

脚本自动激活原环境并配置路径，不安装依赖。Ctrl+B再D脱离tmux；断网不影响该本地队列，关机/重启/休眠会中断。查看：

```bash
cat "$HOME/v12_linux/outputs/stage0_3_20261107/REPORT_ZH.md"
cat "$HOME/v12_linux/outputs/stage0_3_20261107/STAGE0_3_COMPLETED.json"
```

如需简短回传完成摘要，避免复制长哈希清单被终端截断：

```bash
python3 - <<'PY'
import json
from pathlib import Path
d=json.loads((Path.home()/"v12_linux/outputs/stage0_3_20261107/STAGE0_3_COMPLETED.json").read_text())
print(json.dumps({k:v for k,v in d.items() if k!="evidence_sha256"},ensure_ascii=False,indent=2))
PY
```

技术依据：Linux内核cgroup v2与/proc官方文档 https://docs.kernel.org/admin-guide/cgroup-v2.html 、https://docs.kernel.org/filesystems/proc.html ；nproc环境变量语义 https://www.gnu.org/s/coreutils/manual/html_node/nproc-invocation.html 。这些文档用于监控实现，不修改服务器cgroup或系统配置。
