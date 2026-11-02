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
        correction=settings()['correction'],automatic_authorized_Stage1_next=True,time=time.time())
    r.write(OUT/'STAGE0_RESULT.json',result)
    lines=['# V11阶段0：源动作策略通道实现与一致性检查','',
        '用户更正：原文 −log π 作废。实际采用 O1 的优势加权动作 MSE，exp(10A) 截断上限100。',
        'Lπ=目标加权MSE均值+w_a×源加权MSE均值；动作维先求均值。源/目标优势均取同一步V更新前的 min(target Q1,target Q2)−V。',
        'SA-all对64条源样本求均值。SA-top先取源批次优势前16/64，再要求A>0，未入选权重为0；按v11表格对入选样本求均值，无入选时源项0。优势并列按批内原位置优先。',
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
        '## 后续预算与接续','']
    for k,a in estimates.items():lines.append('- %s：单次500k约%.2f小时，10seed约%.2f小时。'%(k,a['estimated_single500k_hours'],a['estimated_10seeds_hours']))
    lines+=['','20项直接外推合计约%.2f小时，另需逐项CPU审计/最终核验；短前缀与后期episode长度可能不同，非精确承诺。'%budget['stage1_20jobs_hours'],
        '按用户最新授权，阶段0核验通过后直接运行阶段1，SA-all201–210后SA-top201–210，阶段1完成停下。阶段2Walker上限、V9阶段4、T调参均不启动。',
        '原始指令快照及更正见config.json、authorization.json；命令/模型配置/代码数据XML哈希见launch1.json、FROZEN.json、runs/*/config.json。','']
    with open(r.winpath(OUT/'STAGE0_REPORT_ZH.md'),'x',encoding='utf-8') as f:f.write('\n'.join(lines))
    r.write(OUT/'ANALYSIS_COMPLETED.json',dict(passed=True,CPU_only=True,result_sha256=r.sha(OUT/'STAGE0_RESULT.json'),
        report_sha256=r.sha(OUT/'STAGE0_REPORT_ZH.md'),time=time.time()))

if __name__=='__main__':main()
