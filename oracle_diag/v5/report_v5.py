"""Chinese delivery from independently audited scalars; refuse incomplete final reports."""
from audit_v5 import *

def link(path,label=None):
    path=Path(path).resolve();return '[%s](<%s>)'%(label or path.name,path.as_posix())

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('preparation','interim','final'),required=True);a=p.parse_args()
    protocol=read(OUT/'protocol.json');A=read(OUT/'cpu_a1/completed.json');B=read(B_OUT/'cpu_b3/completed.json');diagnostic=read(OUT/'a5_2/completed.json')
    assert A['passed'] and B['passed'] and diagnostic['cuda_initialized'] is False
    protected=protected_check();baseline=read(OUT/'existing_baselines.json');rows=list(baseline['rows']);comparisons={}
    for r in rows:assert sha(r['tb_file'])==r['tb_sha256']
    jobs=[j for j in read(OUT/'training_plan.json')['jobs'] if j['kind']=='formal']
    if a.stage!='preparation':
        assert read(OUT/'smoke_audited.json')['passed']
        required=jobs if a.stage=='final' else [j for j in jobs if j['method'] in (A_METHODS[0],B_METHODS[0])]
        rows.extend(audit_job(j) for j in required)
        for stack in ('A','B'):
            available={r['method'] for r in rows if r['stack']==stack}
            for key,(x,y) in protocol[stack]['comparisons'].items():
                if x not in available or y not in available:continue
                left={r['seed']:r['last5_mean'] for r in rows if r['stack']==stack and r['method']==x}
                right={r['seed']:r['last5_mean'] for r in rows if r['stack']==stack and r['method']==y}
                seeds=protocol[stack]['seeds'];assert sorted(left)==sorted(right)==seeds
                comparisons[key]=dict(comparison=[x,y],**paired([left[s]-right[s] for s in seeds]))
        if a.stage=='final':assert len(comparisons)==8
    title={'preparation':'CPU 准备与零训练诊断交付（待 GPU 确认）','interim':'中期结果','final':'最终结果'}[a.stage]
    lines=['# Oracle v5：'+title,'','## 状态','']
    if a.stage=='preparation':lines+=['A、B 实现和 CPU 检查完成；**v5 尚未运行 GPU 冒烟或正式训练，不能据此回答 QA1–QA4、QB1–QB4。**',
      '已准备 30 项正式训练（A 15 项、B 15 项）、8 项单独 GPU 冒烟及 2 项并行冒烟。已有基线仅重读并核验日志，跨入口复用还须通过 GPU 检查。','']
    else:lines+=['本报告由原始 TensorBoard 重算；每个 seed 内先取固定末五次评估均值，再以 seed 为独立单位计算配对差。','']
    lines+=['## 1. 实现与证据','',
      '- A：CRITIC-ORACLE / CRITIC-SOURCE 仅在 actor 中排除 source；Q-ORACLE 还在 V 中排除 source。每步仍抽样 64+64，source 系数 1.0。',
      '- A：固定网络和 target batch、更换 source batch 后，前两组 actor 逐位一致；Q-ORACLE 的 actor、V 逐位一致；应受影响的网络确实改变。独立梯度检查也确认被排除的 source 项无梯度。',
      '- A：v5 的 ORACLE-W1 复用入口与 v4 在 CPU 的 10 步参数、损失轨迹和抽样计数完全一致；GPU 复用检查单独执行。',
      '- B：TargetIQL 只覆盖 train；V、Q、actor、目标 Q 更新、VAE 和学习率调度均沿用官方实现。IQL-TARGET 与 +REG 仅用官方 noreg 开关区分，source 训练抽样为 0，target 为 128。',
      '- B：OTDF-NOREG 使用官方类，CPU 检查确认抽样 128+128、过滤后 source 为 102，三个损失的有效 batch 均为 230。Q 使用 OT 权重，V 和 actor 不使用该权重。',
      '- B：保留官方 actor 中优势 [B,1] 与 BC 损失 [B] 的广播行为，未在消融中悄悄修复；因此结果解释限定于该官方实现。',
      '- 数据：A 使用 5000 条 explicit_next_v3 target；B 保留官方 4999 条 next-row target；B 原始数据、OT 代价、导出行次序和代码通过历史哈希核验。',
      '- 旧文件保护：本次完整核验 %d 项哈希，全部一致。'%protected,'',
      '证据：'+', '.join(link(x) for x in [OUT/'cpu_a1/completed.json',B_OUT/'cpu_b3/completed.json',OUT/'existing_baselines.json',OUT/'protocol.json'])+'。','',
      '## 2. A5：固定 5000 条 target 状态的动作诊断','',
      'MSE 在全部状态与 6 个动作维度上平均。source 参考动作来自 target 均值/标准差（总体 SD + 1e-6）标准化空间的精确 5 近邻，再平均近邻动作；12 个预先固定查询点通过独立穷举距离核对。表中为五种子均值 ± 样本 SD。','',
      '| 策略 | 与 target 动作的 MSE ↓ | 与近邻 source 均值动作的 MSE ↓ |','|---|---:|---:|']
    for method,x in diagnostic['summary'].items():
        t=x['target_action_mse'];s=x['source_neighbor_action_mse']
        lines.append('| %s | %.8f ± %.8f | %.8f ± %.8f |'%(method,t['mean'],t['sample_sd'],s['mean'],s['sample_sd']))
    lines+=['', '两个含 source 的策略都更远离 target 动作、更接近近邻 source 动作。ORACLE-W1 向 source 靠近的绝对幅度较小（MSE 约 0.42354→0.41957）；不能把相对倍数或 source-only 的大幅差异外推为混合策略的主要因果机制。',
      '这是描述性支持，不能代替路径切断后的回报对照；近邻 source 动作是代理参照，并非 shifted 环境的最优动作。',
      '原始逐 seed 记录、模型路径及哈希：'+link(OUT/'a5_2/completed.json')+'。','',
      '## 3. 分组与逐 seed 结果','',
      '**A 与 B 的分数不得直接排名比较。** A 为 500k 步、seeds 0–4；B 为 1M 步、seeds 1–5，且 loader、网络更新实现和 batch 不同。B 有正则时另预训练 VAE 10k 步；无正则组不执行 VAE 预训练。','']
    if a.stage=='preparation':lines+=['以下仅为已存在基线的重新核验值，不是 v5 新训练结果。','']
    for stack in ('A','B'):
        lines+=['### 实验 '+stack,'','| 方法 | seed | Last-5 | Final |','|---|---:|---:|---:|']
        for r in rows:
            if r['stack']==stack:lines.append('| %s | %d | %.6f | %.6f |'%(r['method'],r['seed'],r['last5_mean'],r['final_score']))
        lines.append('')
    lines+=['## 4. 预设比较','',
      '通过条件：五种子配对差均值 > 2×SE 且至少 4 个正号，SD 使用 ddof=1，SE=SD/√5。这是工程门槛，不是严格 95% 置信结论。','',
      '| 编号 | 比较（左 − 右） | 配对差 seed 顺序 | 均值 | SD | SE | 正号数 | 门槛 |','|---|---|---|---:|---:|---:|---:|---|']
    for stack in ('A','B'):
        for key,pair in protocol[stack]['comparisons'].items():
            if key not in comparisons:lines.append('| %s | %s | 待训练 | — | — | — | — | 待定 |'%(key,' − '.join(pair)));continue
            x=comparisons[key];lines.append('| %s | %s | %s | %.6f | %.6f | %.6f | %d/5 | %s |'%(key,' − '.join(pair),', '.join('%.6f'%z for z in x['values']),x['mean'],x['sample_sd'],x['se'],x['positive'],'通过' if x['passed'] else '未观察到稳定正向差异'))
    lines+=['','## 5. 结论的解释范围','',
      '- QA1 检验当前 critic-only source 使用方案相对 TARGET-64 的回报；QA2 检验移除 source 直接 actor 损失的效果，不单独证明 actor 是唯一或主要伤害来源。source 仍可通过学得的 Q、V 间接影响后续 actor。',
      '- QA3 衡量这一特权标签修正干预的实际增益，不是动力学修正的数学上限；QA4 检验再移除 source V 损失的影响。',
      '- QB1 是同一官方代码中的 OTDF 与 target-only 比较；QB2 衡量过滤加权 source 及 batch 从 128 增到 230 的组合干预，不是严格固定 batch 下的纯数据效应。',
      '- QB3、QB4 检验正则流程的效果；VAE 预训练会消耗随机数，同 seed 不意味着之后抽样序列相同。通过也不能宣称该正则在任何方法中都必要。',
      '- 不通过正向门槛只能报告未观察到稳定正向差异，不能推断等效、所有源数据无价值或所有源数据有害，也不能证明优于论文方法。',
      '- 局限：单任务；target 仅约 8 条轨迹段；V 是学得函数；oracle 利用历史恢复的隐藏物理状态；历史复用与新实验运行时间不同；Windows 兼容复现的软件环境与论文有差异。保留阶段 A 失败和历史恢复证据。','']
    if a.stage!='preparation':
        lines+=['### 已完成比较的直接读数','']
        for key,x in comparisons.items():lines.append('- %s：%s，配对均值 %.4f，SE %.4f，正号 %d/5。'%(key,'观察到达到工程门槛的正向差异' if x['passed'] else '未观察到达到工程门槛的正向差异',x['mean'],x['se'],x['positive']))
    if a.stage=='preparation':
        lines+=['## 6. 下一阶段与确认点','',
          '先执行两条管线各自的复用冒烟和三个新组冒烟，再执行 A/B 并行吞吐冒烟。每个进程限制 2 个 CPU 线程。测得吞吐后给出两条正式队列的预计耗时；如 B 超过 72 小时，将列出可选缩减方案，由用户决定。当前完整 30 项计划保持不变。',
          'GPU 冒烟及正式训练均未启动。用户文件要求 GPU 前确认，并要求冒烟汇报后再次确认正式训练。',
          '若正式并行时发现吞吐低于单独运行的一半，脚本将阻止 B 启动下一项，优先完成 A 前两组；正在运行的 B 任务保留到安全作业边界，避免丢失不可续训的优化器状态。该操作方式会在冒烟汇报中明确列出。',
          '准备阶段修复：A5 使用旧 SciPy 支持的精确单线程近邻查询；B 模块改用独立名称以防误导入 A；旧 v3 评估凭据没有 time_limit 字段，历史审计按其已有实际环境标识核验，新运行仍强制检查 1000 步限制。所有原始训练、完成标记和失败训练产物均未改变。','']
    lines+=['## 附录：逐 checkpoint 与来源','',
      '| 实验 | 方法 | seed | Last-5 五个原始归一化分数 | TensorBoard 来源 |','|---|---|---:|---|---|']
    for r in rows:lines.append('| %s | %s | %d | %s | %s |'%(r['stack'],r['method'],r['seed'],', '.join('%.6f'%v for v in r['last5']),link(r['tb_file'],'原始日志')))
    filename={'preparation':'PREPARATION_v5.md','interim':'INTERIM_v5.md','final':'REPORT_v5.md'}[a.stage]
    path=HERE/filename
    with open(path,'x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    write(OUT/(a.stage+'_report_evidence.json'),dict(stage=a.stage,report=str(path),report_sha256=sha(path),protected_files=protected,rows=rows,comparisons=comparisons,
      verification='Original TensorBoard re-read, sample SD and paired differences independently recomputed'))
    print(str(path))

if __name__=='__main__':main()
