# -*- coding: utf-8 -*-
"""Add reviewed interpretation and source links to the reproducible v2 report."""
from support import OUT, HERE, read
from pathlib import Path


def link(label, path):
    return '[' + label + '](<' + Path(path).as_posix() + '>)'


def enrich(text):
    audit = read(OUT/'final_validation.json')
    assert audit['passed'] and audit['training_jobs'] == 15
    attempt = Path(audit['attempt_directory'])
    evidence = read(attempt/'independent_statistics.json')
    primary, const_target, backup_target = evidence['paired']
    methods = evidence['method_summaries']
    assert not primary['passed'] and primary['positive'] == 3
    lead = '''## 结论摘要

**A–D 已完成并通过独立验收；本轮结果不支持继续扩大当前设定下的修正或加权训练。** E 按要求只估算工作量，未启动。

- **Oracle 修正未稳定优于常数权重。** BACKUP−CONST 的五种子均值为 **{gain:+.2f} 分**，标准误 **{se:.2f} 分**，只有 **{positive}/5** 个种子为正；均值没有超过 **2×SE={twose:.2f}**，也未达到至少 4/5 为正的门槛。
- **当前 source 混入方式没有帮助 target-only 基线。** CONST−TARGET 平均 **{const:+.2f} 分**，五个种子全部为负。BACKUP−TARGET 平均 **{bt:+.2f} 分**，也没有建立优势。这里 TARGET 每步使用 256 条 target，迁移组使用 128 source＋128 target，因此这是等总采样量下的实际方案比较，不能单独证明 source 数据本身无用。
- **没有修正器通过预设离线门槛。** 原 Walker 的 A、B、M、M-gate 均未在两个固定 V 下同时将 Bellman 误差的均值和中位数降低至少 20%；四个新任务的补充评测同样没有通过者。因此 D 按约定只运行 TARGET、CONST、ORACLE-BACKUP。
- **可确认的工程改进是 source loader 修复。** 三环境均由 4999 行恢复为 5000 行；分别修复 4、12、4 条跨轨迹边界的下一状态，并各补回末行。新旧 loader 的成绩保持分表。

上述结论只适用于本轮 Walker-kinematic、5000 条 source、10% 正常动力学 target、固定权重 0.20 与 IQL 的设置。**未过门槛意味着当前证据不足，不是证明所有动力学修正都无效。**

'''.format(gain=primary['mean'], se=primary['se'], positive=primary['positive'], twose=2*primary['se'], const=const_target['mean'], bt=backup_target['mean'])
    text = text.replace('所有阶段完成，以下为 fixed_v2 loader 的正式五种子结果。',
                        'A–D、四个新任务的补充 B 及最终验收均已完成；E 仅估算工作量。')
    text = text.replace('## 1. 审查后的协议', lead+'## 1. 审查后的协议', 1)
    protocol = '''### 本轮具体比较了什么

source 为 shifted Walker2d-kinematic-footjnt 的 medium 数据；target 为正常动力学 D4RL medium 的 10% 子集，评估也在正常动力学 target 环境。两个方向不能对调解释。

- TARGET：只用 target 更新原 IQL。
- CONST：source 权重固定为 0.20，source Q 目标为 `r_s + 0.99 × nd_s × V(s'_s)`。
- ORACLE-BACKUP（表中简称 BACKUP）：source 权重仍为 0.20，仅将 source Q 目标替换为 `r° + 0.99 × nd° × V(s'°)`；`(r°, s'°, nd°)` 来自同一个 source `(s,a)` 在 target 模拟器下的验收转移。V 与 actor 的 source 状态、动作及训练流程仍沿用原方法。

`nd=1` 表示允许 bootstrap，`nd=0` 表示终止。训练中的 V 持续更新；B 阶段使用冻结的旧 TARGET/MIX_FB V 做离线诊断。这两种 V 的用途不同。

本轮检验的是 **Bellman 目标修正**，没有扩大 OT 训练，也没有与 TABB/OTDF 做同协议成绩比较，因此不能据此声称超越它们。`eta=1` 未调整；这里采用常数 source 权重，没有把失配重新送入指数降权公式。

'''
    text = text.replace('## 2. 阶段完成情况', protocol+'## 2. 阶段完成情况', 1)
    bnote = '''### 为什么状态误差变小，仍不够支持训练

M 的标准化状态误差均值由 C 的 1.6188 降至 1.4900，但两个固定 V 下的 Bellman 误差均值反而增加约 42.4% 和 36.4%。这是本轮实测事实。合理解释是：状态空间中较小的平均距离，不保证价值函数敏感方向、奖励和终止掩码上的误差也较小；本轮没有单独识别这几个成分的因果贡献。

M 在 target 验证集的状态误差均值为 0.3256，而在 source 查询点上为 1.4900，提示 source 外推更困难，但两组状态分布不同，不能把差值全部归因于分布外误差。M-gate 只在 6.86% 的 source 点上采用模型，因而主要退回 C；相关性 0.6191 并不等于其阈值已足以筛出能改善 Bellman 目标的点。

'''
    text = text.replace('## 7. 阶段 C：', bnote+'## 7. 阶段 C：', 1)
    extra = read(OUT/'resume1/stage_b_new_tasks/completed.json')['tasks']
    supplement = ['### 补充 B 的诊断与门槛', '',
        '| 任务 | target 验证状态误差均值 | gate 采用 M 比例 | 分歧与 source 状态误差 Spearman | 通过全部 Bellman 门槛的候选 |',
        '|---|---:|---:|---:|---|']
    for task, row in extra.items():
        supplement.append('| {} | {:.5g} | {:.2%} | {:.4f} | 无 |'.format(task,row['target_validation_state_error']['mean'],row['gate_fraction'],row['disagreement_oracle_error_spearman']))
    supplement += ['', '四个任务复用同环境模型；表中的 target 验证误差不是四次独立模型实验。补充 B 只作离线评测，未启动新任务 RL。M-gate 在 Hopper-morph 的状态误差均值有所下降，但 MIX_FB 的 Bellman 误差均值仍变差；不能凭状态误差一个指标放行。', '',
                   '完整补充指标（含中位数、p95、相对 C 改善比例）见附录 B。', '']
    text = text.replace('## 9. 阶段 D：', '\n'.join(supplement)+'\n## 9. 阶段 D：', 1)
    dnotes = ['### 跨种子的汇总与解释', '',
              '| 方法 | 五种子 Last-5 均值 | 种子间样本 SD | 均值 SE |', '|---|---:|---:|---:|']
    for method, item in methods.items():
        dnotes.append('| {} | {:.3f} | {:.3f} | {:.3f} |'.format(method,item['mean'],item['sd'],item['se']))
    dnotes += ['', '**配对差的单位是 normalized score 分数点，不是相对百分比。** SD 描述五个种子的差异，SE=SD/√5；每个种子的五个相邻 checkpoint 先平均为一个数，不能当作 25 个独立样本。', '',
        '主比较的差值从 −11.624 到 +12.491 分，seed2 明显为负，seed0 接近零，其余三个为正。保留全部预设种子，没有剔除负结果或改取最佳 checkpoint。当前结论由完整五种子表决定。', '',
        '补充描述性比较 BACKUP−TARGET：五个配对差为 '+', '.join('{:+.3f}'.format(x) for x in backup_target['delta'])+' 分；均值 {:+.3f}、SD {:.3f}、SE {:.3f}，3/5 为正。它用于说明实际基线位置，不替代预设主比较。'.format(backup_target['mean'],backup_target['sd'],backup_target['se']), '',
        '**决策：结束当前设置下的继续开发与扩训。** Oracle 给出了本轮可获得的更准确一步 target 转移，但没有通过稳定提升门槛；非模拟器修正器也未通过离线门槛。当前不足以支持继续为这组设置增加 OT、模型容量或调权参数。Oracle 的回报不是数学上限，这个结论也不能排除其他数据方向、target 比例或训练方法的价值。', '',
        'loader 修复对训练成绩的独立影响尚未做同配置、同随机流的五种子消融；v2 同时统一了首次训练更新的随机流，所以不能把 v1→v2 的成绩变化全部归于 loader。', '']
    text = text.replace('## 10. 阶段 E', '\n'.join(dnotes)+'\n## 10. 阶段 E',1)
    cost = '\n本轮 15 项正式训练实测合计 **{:.2f} 小时**，从首项启动到末项结束约 **{:.2f} 小时**。若仅将 target 比例改为 1%，仍做 3 方法×5 seeds×500k，可先按约 **38–45 小时单 GPU** 规划；这是基于当前机器吞吐的估计，数据比例及后台负载会改变耗时。若改变数据方向，还需额外的加载、模拟器、评估环境和 oracle 对齐验收，不能只按 15 项训练计费。阶段 E 需要用户另行决定。\n\n'.format(audit['total_training_hours'],audit['elapsed_queue_hours'])
    text = text.replace('限制：回放为经过阈值验证的近似状态恢复；',cost+'限制：回放为经过阈值验证的近似状态恢复；',1)
    # Replace the old progress-oriented provenance section with verified final sources.
    text = text.split('## 11. 证据与最终交付')[0]
    sources = [('A：三环境 loader 与原运行审计',OUT/'stage_a/completed.json'),
               ('B：Walker 修正器评测',OUT/'stage_b/completed.json'),
               ('C：四任务回放验收',OUT/'stage_c/completed.json'),
               ('补充 B：四任务评测',OUT/'resume1/stage_b_new_tasks/completed.json'),
               ('D：15 项正式队列',OUT/'resume1/queue_status.json'),
               ('独立统计：原始事件、Last-5 与配对差',attempt/'independent_statistics.json'),
               ('独立验收记录',attempt/'audit.json'),
               ('180 个模型和快照的哈希清单',attempt/'model_manifest.json'),
               ('受保护文件及预设判据',OUT/'preregistered_protocol.json'),
               ('长路径错误及恢复证据',attempt/'recovery_provenance.json')]
    text += '## 11. 最终验收、来源与复现\n\n'
    text += '- 15/15 项训练均正常退出并到达 500k；每项含 50 个预设 target 评估点，Last-5 均由原始 TensorBoard protobuf 独立读取。\n'
    text += '- 180 个模型/优化器/调度器文件与 V 快照全部在 CPU 上加载并验证数值有限；最终 V 与 500k V 快照逐张量一致。\n'
    text += '- 30 个受保护旧文件、4 个冻结训练文件、14 个保留的重启前补充产物哈希不变；每个 seed 的初始化与 target 子集一致，CONST/BACKUP 首批索引一致，source/oracle 与 HDF5 全量逐行对齐。\n'
    text += '- 75 组诊断误差数组及旧版 15 次运行的统计完成复核。Walker/Hopper 的模型训练与验证轨迹无交集。诊断与最终验收均未初始化 CUDA。\n'
    text += '- 原队列在训练全部结束后，因 Windows 260 字符路径限制读取调度器文件失败。文件实际存在；本次在新的验收目录使用 `\\\\?\\` 长路径前缀完成读取。原错误、状态和报告草稿已保留，没有重跑训练或更改科学设置。\n\n'
    text += '\n'.join('- '+link(label,path) for label,path in sources)+'\n\n'
    text += '复算命令（`--attempt` 必须使用新的空目录，已有目录拒绝覆盖）：\n\n```powershell\n'
    text += "& 'D:\\ANACONDA\\envs\\offdynamics\\python.exe' -B -u '"+str(HERE/'audit_completed_v2.py')+"' --attempt '"+str(OUT/'final_attempt_NEW')+"'\n```\n\n"
    text += '重建本报告：\n\n```powershell\n'
    text += "& 'D:\\ANACONDA\\envs\\offdynamics\\python.exe' -B '"+str(HERE/'report_resume_v2.py')+"'\n```\n\n"
    text += '### 对最初三个问题的回答\n\n'
    text += '**loader** 修复了跨轨迹下一状态的错误连接和末行丢失，涉及三环境原有 4/12/4 行并各补 1 行。**非模拟器修正器**在本轮测试范围内没有一个达到预设的 Bellman 目标准确度门槛。**Oracle 修正目标**在五种子下平均比 CONST 高 4.399 分，但波动大、仅 3/5 为正，未建立稳定优势；当前停止扩训，E 的数据设定变更留待用户决定。\n\n'
    text += '## 附录 A：15 次运行的原始 Last-5\n\n| seed | 方法 | 460k | 470k | 480k | 490k | 500k | 平均 |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for run in evidence['runs']:
        text += '| {} | {} | {} | {:.6f} |\n'.format(run['seed'],run['method'],' | '.join('{:.6f}'.format(v) for v in run['last5']),run['mean'])
    text += '\n表中保留六位小数便于核对，不表示测量精度达到六位；TensorBoard 的 scalar 为 float32，精确存储值和事件文件路径见独立统计 JSON。\n\n'
    text += '## 附录 B：新任务修正器补充指标\n\n状态误差在 target 标准化空间计算；Bellman 相对改善为 `1 − error/C_error`，负数表示变差。C 的原始均值和中位数已在第 8 节逐行并列。\n\n'
    text += '| 任务 | 方法 | 状态均值 / 中位 / p95 | 状态比 C 更准比例 | TARGET Bellman 均值 / 中位改善 | MIX_FB Bellman 均值 / 中位改善 |\n|---|---|---|---:|---|---|\n'
    for task, row in extra.items():
        for key, item in row['metrics'].items():
            vals=['{:+.2%} / {:+.2%}'.format(item['bellman'][kind]['mean_relative_improvement'],item['bellman'][kind]['median_relative_improvement']) for kind in ['TARGET','MIX_FB']]
            text += '| {} | {} | {} | {:.2%} | {} | {} |\n'.format(task,key,' / '.join('{:.5g}'.format(item['state'][k]) for k in ['mean','median','p95']),item['better_than_C_fraction'],*vals)
    return text+'\n'
