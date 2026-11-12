"""Descriptive V14 report from completed CPU diagnostics, with no new proposals."""
from core import read,stats


def fmt(value):
    if value is None:
        return '无样本'
    if isinstance(value,float):
        return f'{value:.7g}'
    return str(value)


def table(headers,rows):
    return ['|'+'|'.join(headers)+'|','|'+'|'.join('---' for _ in headers)+'|']+[
        '|'+'|'.join(fmt(v).replace('|','/') for v in row)+'|' for row in rows]


def build(out,inputs,a,b,c,resources):
    lines = ['# V14 Hopper 动作修正失败原因诊断报告','',
        '仅 CPU：新训练更新 0、优化器实例 0、GPU 作业 0。A3 是获授权的新诊断评估，B 是仿真跟踪，不计入 V13 正式结果。',
        f'入口：`bash ~/v14_diag/run_in_tmux.sh`；本次输出：`{out}`。',
        '新代码、配置、逐步记录和图表只写入 v14_diag；V13 模型、缓存、日志及原环境使用前后哈希核对。',
        '固定源样本、最近邻参考样本、源完整轨迹编号和全部参数见 config.json。求解器、容差、有限差分步长和归一化尺度直接读取并校验 V13。','',
        '## A：最终 BC 模型、训练记录与实际轨迹','',
        '模型：BC-all、BC-fix、BC-mix、BC-mix-sel，种子 601–605，均为 V13 最终 200k 模型；CPU 推理，无优化器。',
        'A1 在全部目标 5,000 行与固定无放回 20,000 条修正源样本上计算动作维均值后样本均值 MSE。分档使用原轨迹内 0 起始步数。',
        'A1 的误差累积用 float64；原 GPU 训练日志为 float32。A2 是末尾 5,000 次、每次更新前不同模型和随机批次的损失，A1 是最终模型的固定数据误差，不能要求逐值相等。','']
    lines += table(['组别','目标 MSE（五种子均值）','目标 MSE 样本 SD','修正源 MSE（五种子均值）'],[
        (name,r['target_MSE_mean'],r['target_MSE_seed_SD'],r['source_MSE_mean']) for name,r in a['groups'].items()])
    lines += ['',f'BC-mix / BC-all 的目标 MSE 均值比：{fmt(a["mix_to_all_MSE_ratio"])}。',a['interpretation'],'',
              '### A1 逐模型结果','']
    lines += table(['组别','种子','目标 MSE','修正源 MSE'],[
        (r['variant'],r['seed'],r['target_MSE'],r['corrected_source_MSE']) for r in a['A1']])
    lines += ['','### A2 训练日志核对','']
    lines += table(['种子','末 5000 目标项','末 5000 源项','最终模型目标全数据 MSE','最终模型源样本 MSE'],[
        (r['seed'],r['target_loss_mean'],r['source_loss_mean'],r['final_target_full_MSE'],r['final_source_sample_MSE']) for r in a['A2']])
    lines += ['',
        '核查依据：原 worker.py 的 one_update 先采样 64 条目标、再采样 64 条源，分别计算 MSE 均值相加后执行一次 Adam / cosine。',
        '本次重建两个独立 RandomState 的完整 200,000×64 行采样流，核对摘要与原 COMPLETED.json 完全一致；核对完整 steps.f32 200,000 行且目标项+源项逐位等于总项。',
        '已有 AUDIT.json 核对最终 Adam / cosine 与完整日志，本次只读引用其证据，不再执行优化更新。代码与采样依据、每个流摘要保存于 A/A2_log_check.json。','',
        '### A3 新 CPU 轨迹诊断','',
        '各最终模型评估 10 回合，共 200 回合，种子 500000 + 10×(训练种子−301) + episode；XML / 1000 步上限同 V13。记录每一步当前观测、动作、下一观测、奖励与终止。',
        '最近邻距离为按 V13 目标观测样本标准差加 1e-6 缩放的 L2；目标参考集 5,000 行、源参考集固定 100,000 行，集合大小不同，距离只作描述。',
        '每模型／种子／步数分档的全表：A/A3_neighbor_bins.csv。BC-mix 到达状态上三个同种子模型两两动作距离：A/A3_actions_on_mix_states.csv。逐回合长度与回报：A/A3_episodes.csv；逐步原始值：A/*.npz。',
        '下表将五种子各 10 回合的实际状态按组和步数合并；n 为该档真实达到的状态数，晚期档只包含存活到该档的回合。','']
    if 'A3_pooled' in a:
        lines += table(['组别','步数档','n','目标 NN 均值','源 NN 均值','源 NN 更近比例','mix-all 动作距离','mix-fix 动作距离','all-fix 动作距离','mix动作更近fix比例'],[
            (r['variant'],r['step_bin'],r['n'],r['target_NN_mean'],r['source_NN_mean'],r['source_NN_closer_fraction'],r['mix_all_L2'],r['mix_fix_L2'],r['all_fix_L2'],r['mix_action_closer_to_fix_fraction']) for r in a['A3_pooled']])
    lines += ['','A3 不预设“离开区域”的额外距离阈值，不将最近邻距离差直接当成因果证据。结合上表早期档的两域距离与三模型动作距离描述，完整分布见对应 CSV。','',
              '## B：目标自检与源轨迹跟踪','',
              f'目标 B0 包含全部 {len(b["B0"]["trajectories"])} 个记录段，包括末尾不足整回合的片段。每步原始下一观测最大绝对误差门槛直接读取 V13 config：{fmt(b["B0"]["raw_linf_limit"])}。',
              'B0 要求全部记录步跟踪完成；任何中途终止不通过，末步终止标签须与记录一致。原始误差和归一化残差分别保存。','']
    lines += table(['轨迹','L','T','通过','首次超限步（1起）','误差最大值','误差中位数','误差P90','动作差均值','动作差最大值'],[
        (r['trajectory_id'],r['L'],r['T'],r['self_check']['passed'],r['self_check']['first_exceed_step'] if r['self_check']['first_exceed_step'] is not None else ('未超限' if r['T'] else '未能恢复'),
         r['raw_linf']['max'],r['raw_linf']['median'],r['raw_linf']['p90'],r['action_delta']['mean'],r['action_delta']['max'])
        for r in b['B0']['trajectories']])
    lines += ['','B0 结论：'+('通过，按授权自动执行 B1/B2。' if b['B0']['passed'] else '不通过。B1/B2 未启动；A、C 按用户补充授权完成。未放宽门槛。'),
              f'可识别完整源轨迹 {inputs.complete_source_count} 条，本次固定种子无放回选取 {len(inputs.selected_source)} 条。',
              '完整源轨迹要求起点有已记录的上一 terminal/timeout 边界、终点为 terminal/timeout，且中间不跨显式 next 的断点；未知数据头尾排除。缓存 C 保留全部 canonical 行。',
              'B1 顺次执行原源动作。B2 每步从仿真器当前实际 qpos/qvel 求解，候选均复原到该实际状态；选定动作再单独执行并将实际状态带到下一步。仅起点使用源状态。',
              '候选仿真完全沿用 V13 / V9 的 sim.reset 与 qacc_warmstart=0 约定；奖励只累计实际执行选定动作的结果，不累计优化试算。','']
    lines += table(['模式','轨迹数','T均值','T中位数','T P10','T P90','T/L均值','T/L中位数','T/L P10','T/L P90','T<200比例','T/L>=.8比例'],[
        (mode,r['episodes'],r['survival_steps']['mean'],r['survival_steps']['median'],r['survival_steps']['p10'],r['survival_steps']['p90'],
         r['T_over_L']['mean'],r['T_over_L']['median'],r['T_over_L']['p10'],r['T_over_L']['p90'],r['fraction_T_under_200'],r['fraction_T_over_L_at_least_08'])
        for mode,r in b['groups'].items()])
    lines += ['','### 跟踪累计回报归一化分数','']
    lines += table(['模式','均值','中位数','P10','P90','最小','最大'],[
        (mode,*[r['normalized_score'][k] for k in ('mean','median','p10','p90','min','max')]) for mode,r in b['groups'].items()])
    lines += ['','### 跟踪残差按步数分档','']
    lines += table(['模式','步数档','n','均值','中位数','P90','最大'],[
        (r['mode'],r['step_bin'],r['n'],r['mean'],r['median'],r['p90'],r['max']) for r in b['residual_bins']])
    if 'B2' in b['groups']:
        lines += ['',b['groups']['B2']['interpretation']]
    lines += ['',
        '局限：B2 是一步贪心跟踪；不可跟踪结论仅限于这种方式，不能排除带前瞻控制器更好。各轨迹累计回报按各自记录长度结束，不是正式策略回报比较。',
        '20 步等宽终止直方图：B/B_physical_termination_histogram.svg；全分箱计数见同名 CSV。物理终止作为摔倒代理，含记录末步自然终止；记录长度耗尽而没有物理终止的不计入。',
        'B0 无论通过与否均保留逐步 NPZ、逐轨迹 JSON；B0 原始误差完整分布和动作差分布见 B0_RESULT.json。B/B_trajectories.csv 保留每条 L、T、终止、触边比例及回报。','',
        '## C：残差在源轨迹中的位置','',
        '按 raw row_id 还原源轨迹内步数，0–399 每 20 步一档、>=400 合一档。只读取 V13 缓存，不产生仿真步。','']
    lines += table(['步数档','n','e0均值','e0中位数','e1均值','e1中位数','触边比例','e1<=.0840保留比例'],[
        tuple(r.values()) for r in c['bins']])
    lines += ['','## 资源、保护核对和停点','',
        f'资源盘点：nproc={resources["limits"]["nproc"]}，有效可用 CPU={resources["limits"]["effective_visible_cpu_cores"]} 核。',
        f'整机 CPU 占用={fmt(resources["host_CPU_busy_percent"])}%，包含其他用户；总耗时={fmt(resources["elapsed_seconds"])} 秒。',
        f'环境步计数：A={a["environment_steps"]}；B={b["simulator_steps"]}（含构造、恢复、候选试算和实际执行）；C=0。',
        '实际进程并发、RSS 和有效内存逐次记录在 RESOURCE.jsonl / B/CPU_ADMISSION.json；每进程 PyTorch / BLAS 线程数为 1，总 CPU 并行进程不超过 32。',
        '读取前后保护核对见 INPUT_HASHES_BEFORE.json 与 INPUT_PROTECTION_AFTER.json。原环境绑定导入写入路由到本次新 private_runtime，原安装目录不写入。',
        '所有数值必须有限；代码异常、哈希不符或技术自检不通过时留证停止，不自动重试。B0 判据未满足是独立科学停点，只禁止 B1/B2，按最新补充要求 A、C 继续。',
        'AWAITING_USER_REVIEW。本次不提出或启动下一项实验，V13 阶段 2、其他任务和任何训练均未启动。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
