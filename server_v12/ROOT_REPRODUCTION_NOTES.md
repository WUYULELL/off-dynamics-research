# ROOT 来源与复现行为

上游：https://github.com/cristianhtvc/ROOT
固定提交：0ae9c05d70ed369c16732c71abda59b54b559da9。
原始来源清单及各上游文件哈希见 code/stage1_2_20261107/ROOT_PROVENANCE.json。本次不整体复制 upstream_root/，数据文件同样排除。

## 自有入口

- code/stage1_2_20261107/root_env.sh / root_requirements.txt：独立 ROOT 环境，未修改 offdynamics38。
- code/stage1_2_20261107/root_trial.py：阶段 2 核对、权重与短跑。
- code/stage2_continue_20261108/：阶段 2 接续入口。
- code/stage3_20261108/stage3_worker.py 及 shared/shared50/shared60/switch 版本的控制器：正式实验与资源调度；实际文件路径见新增文件清单。

## 按用户事先确认保持的行为

action_dim 从环境读取，metric 沿用命令行默认 euclidean；Q 权重的上游广播方式保留，actor update_policy 调用不传样本权重，proportion 过滤分支仍为注释，未作科学实现修复。
启动前同一真实批次 CPU 核查的 Q 损失：上游广播 23.8276978、逐样本加权 22.1358051、不加权 23.8468361。上游广播形状使损失近似为批内平均权重乘不加权损失，检查不改变训练配置。

阶段 3 源域恢复上游 D4RL 相邻观测读取；目标域使用 V12 显式 next，共 5000 行，与 T/BC 对齐。
源域 513 条终止转移与阶段 2 显式 next 口径不同，Q 的终止掩码屏蔽此差异，OT 特征可能受影响；目标域与上游 loader 有 7 条转移差异，原目标 loader 4999 行。
确认权重计算与训练种子无关后，只计算一次并为五个正式种子共用；文件哈希和两阶段权重分布/相关系数在阶段 3 报告中记录。权重数据本体不上传。
本次为公开代码生效行为的复现，不把广播行为、未加权 actor 或未生效过滤自行改为其他算法。修复版与消融未获授权。
