# -*- coding: utf-8 -*-
"""Stage0 diagnostics and timing only; no efficacy decision from short trials."""
from common import *
from audit import diagnose

def main():
    guard();results={};inputs={}
    for trial_id,variant,wa,steps in TRIALS:
        path=OUT/'runs'/trial_id;assert r.read(path/'audit.json')['passed']
        done=r.read(path/'completed.json')
        diag=diagnose(path,variant,steps,wa)
        results[trial_id]=dict(completed=done,diagnostics=diag)
        for p in sorted(path.rglob('*')):
            if p.is_file():inputs[str(p)]=r.sha(p)
    # Project only the one-model SA trials, not the two-model parity control.
    estimates={}
    for key in ('SA_all_s201','SA_top_s201'):
        d=results[key]['completed'];events=r.read(OUT/'runs'/key/'evaluations.json')['events']
        update=d['training_and_logging_seconds']/20000.*500000.
        evaluation=sum(e['elapsed_seconds'] for e in events)/len(events)*50.
        estimates[key]=dict(estimated_single500k_hours=(update+evaluation)/3600.,
            estimated_10seeds_hours=(update+evaluation)*10/3600.,
            training_logging_seconds_projected=update,evaluation_seconds_projected=evaluation,
            warning='20k prefix extrapolation; runtime/episode-length variation and audit overhead remain')
    budget=dict(single_by_variant=estimates,stage1_20jobs_hours=sum(v['estimated_10seeds_hours'] for v in estimates.values()),
        exact_future_completion_guarantee=False,based_on_short_trials=True,measured_before_formal_Stage1=True,time=time.time())
    r.write(OUT/'MEASURED_BUDGET.json',budget)
    result=dict(passed=True,wa0_parity=r.read(OUT/'runs/wa0_parity_s201/PARITY_RESULT.json'),
        trials=results,budget=budget,input_sha256=inputs,new_GPU_learner_updates=50000,
        original_O1_GPU_reference_updates=10000,total_GPU_learner_updates=60000,total_GPU_Adam_steps=180000,
        CPU_synthetic_learner_updates=8,CPU_synthetic_Adam_steps=24,
        real_environment_episodes=50,Last5_and_H_act_not_applicable_at_20k=True,
        correction=settings()['correction'],automatic_authorized_Stage1_next=False,
        stage1_authorized=False,stage2_authorized=False,repair_evidence=r.read(OUT/'REPAIR_CHANGELOG.json'),time=time.time())
    r.write(OUT/'STAGE0_RESULT.json',result)
    lines=['# V11阶段0：源动作策略通道实现与一致性检查','',
        '用户更正：原文 −log π 作废。实际采用 O1 的优势加权动作 MSE，exp(10A) 截断上限100。',
        'Lπ=目标加权MSE均值+w_a×源加权MSE均值；动作维先求均值。源/目标优势均取同一步V更新前的 min(target Q1,target Q2)−V。',
        'SA-all对64条源样本求均值。SA-top先取源批次优势前16/64，再要求A>0，未入选权重为0；按用户最新澄清对入选样本求均值（分母=入选数），无入选时源项0。优势并列按批内原位置优先。',
        'V/Q与O1相同，w=1，64target+64source；actor复用该source批次，无额外抽样，不构造额外oracle缓存。cosine周期始终500k，短试跑不缩短周期。','',
        'w_a=0：seed201新实现10k与原O1同步10k，逐步三损失及最终四网络、三Adam、cosine完整状态逐位一致，并核验历史O1最初10步损失。',
        'SA-all/SA-top各seed201从0运行20k；这是诊断，不作正式Last5或H_act判定。每10k评估10episode，保存恢复训练RNG。',
        '计算计数：新实现50k+原O1参考10k=60k GPU learner更新（180k Adam步骤）；另有8次CPU合成learner更新（24次Adam步骤），CPU真实数据更新0、CPU环境步0、CPU验收未初始化CUDA。GPU评估共50episode。','',
        '|变体|源A>0比例|目标权重和均值|源有效权重和均值|源入选数均值|目标损失均值|源损失均值|实测秒|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in ('SA_all_s201','SA_top_s201'):
        a=results[k]['diagnostics']['statistics']
        fields=['source_A_positive_fraction','target_weight_sum','source_weight_sum','source_selected','target_actor_term','source_actor_term']
        lines.append('|%s|'%k+'|'.join('%.7g'%a[n]['mean'] for n in fields)+'|%.2f|'%results[k]['completed']['elapsed_seconds'])
    lines+=['','全部逐步权重、MSE、筛选与损失在CPU从原始向量独立复核；权重和未除批大小，另存归一化权重和，二者不混称。损失范围/1000步诊断曲线在STAGE0_RESULT.json，原始逐步向量及日志在runs/。未按回报修改参数，也未以短试跑得分选择变体。','',
        '## 后续预算（仅估计，未授权启动）','']
    for k,a in estimates.items():lines.append('- %s：单次500k约%.2f小时，10seed约%.2f小时。'%(k,a['estimated_single500k_hours'],a['estimated_10seeds_hours']))
    lines+=['','20项直接外推合计约%.2f小时，另需逐项CPU审计/最终核验；短前缀与后期episode长度可能不同，非精确承诺。'%budget['stage1_20jobs_hours'],
        '本次按v11修订版只执行阶段0，完成后队列退出并停下汇报。阶段1/阶段2的任务、变体和先后顺序均须用户之后明确确认。Walker后续BC参照确定为BC-all均值38.85（精确值和收据见config.json）；V9阶段4和T调参未授权。',
        '原始指令快照及更正见config.json、authorization.json；命令/模型配置/代码数据XML哈希见launch1.json、FROZEN.json、runs/*/config.json。','']
    lines+=['','## 无入选样本的边界情况','',
        'SA-top源项=sum(入选exp(10A)×动作MSE)/入选数，再乘w_a。无人入选时源项0，不除以0。']
    for key in ('SA_all_s201','SA_top_s201'):
        d=results[key]['diagnostics']
        lines.append('- %s：无人入选批次%d/%d，占比%.6f。'%(key,d['zero_selected_batches'],d['steps'],d['zero_selected_batch_fraction']))
    changes=r.read(OUT/'REPAIR_CHANGELOG.json')
    lines+=['','## 修复文件、行号与一致性证据','',
        '故障修复本身仅明确O1评估函数的导入路径。另列修订版要求的停止范围、配置/报告/诊断改动。',
        '|用途|文件|本次行号|说明|','|---|---|---|---|']
    for item in changes['items']:
        lines.append('|%s|%s|%s|%s|'%(item['category'],item['file'],','.join(map(str,item['new_lines'])),item['description']))
    lines+=['',
        '训练逻辑：actor.py逐字节等同原CPU验收版本；短试跑训练循环AST、V/Q、三Adam和cosine更新顺序与候选相同。SA-top分母按用户澄清维持候选，非修复时暗改。',
        '随机序列：初始化与Sampler源码未改；同seed初始化和完整target/source索引流与O1独立重建值匹配，actor不额外抽样。',
        '评估协议：明确文件路径加载原冻结O1同一evaluate，函数AST/文件SHA核验；每10k十episode、固定evalseed、shifted XML、.002/1000、原归一化、保存恢复Python/NumPy/Torch/CUDA训练RNG均沿用。',
        '本次额外CPU优化0，复用此前8次合成learner更新/24Adam。原启动失败发生在trial.main之前，原GPU更新/环境步0。旧文件和失败现场保留，证据哈希见REPAIR_CHANGELOG.json及DELIVERY_VERIFIED.json。','']
    with open(r.winpath(OUT/'STAGE0_REPORT_ZH.md'),'x',encoding='utf-8') as f:f.write('\n'.join(lines))
    r.write(OUT/'ANALYSIS_COMPLETED.json',dict(passed=True,CPU_only=True,result_sha256=r.sha(OUT/'STAGE0_RESULT.json'),
        report_sha256=r.sha(OUT/'STAGE0_REPORT_ZH.md'),time=time.time()))

if __name__=='__main__':main()
