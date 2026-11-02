"""Publish the verified GPU smoke report; never authorize formal training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support_v4 import *
from audit_v4 import events


def main():
    audit=read(OUT/'smoke_audited.json');reuse=read(OUT/'target_reuse_verified.json')
    assert audit['passed'] and reuse['passed'] and len(audit['runs'])==5
    assert read(OUT/'queue_status.json')['state']=='awaiting_formal_approval'
    assert not (OUT/'gpu_formal_authorization.json').exists()
    rows=[]
    for run in audit['runs']:
        file,scalars=events(run['out'])
        assert sha(file)==run['tb_sha256']
        assert scalars['test/target_normalized_score']=={5000:run['final_score']}
        progress=read(Path(run['out'])/'progress.json')
        rows.append('| '+ ' | '.join([run['method'],'5000',str(progress['source_draws']),str(progress['target_draws']),
            f"{run['elapsed']:.2f}",f"{run['final_score']:.6f}",'通过'])+' |')
    estimate=audit['estimated_formal_hours_from_smoke']
    early=read(OUT/'launch1/early_target_check.json')
    difference=max(r['max_absolute_tensor_difference'] for r in reuse['model_checks'])
    report=HERE/'GPU_SMOKE_REPORT_20261011.md'
    text=f'''# Oracle v4：GPU冒烟与TARGET复用验收

日期：2026-10-11。**5项GPU冒烟全部通过，旧v3的五种子TARGET可复用。20项正式训练尚未开始，等待用户按v4文档确认。**

## 一、运行结果

各组seed0、5000步，在真实shifted Walker2d、OTDF walker_morph.xml、TimeLimit=1000下评估10回合。

| 组 | 更新步数 | 累计source抽样 | 累计target抽样 | 进度计时秒 | 5000步评估分 | 验收 |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

上述评分只用于确认运行与评估流程，不能作正式性能排名；smoke使用5000步学习率日程，不等于500k正式训练的前5000步。正式主指标仍是460k–500k Last-5。

## 二、TARGET复用检查

- v4入口与历史v3 smoke的初始化SHA256、首批采样索引和有效配置完全一致。
- 5000步后的7个模型、优化器、调度器产物逐项比较，张量最大绝对差为 **{difference:.8g}**。
- 历史首次评估分：{reuse['original_score']:.12f}；本次：{reuse['repeated_score']:.12f}；绝对差：{reuse['score_absolute_difference']:.8g}。
- v3没有保存前几步损失。新生成的冻结v3 CPU前10步重放与v4 CPU逐位一致；新GPU的30个损失值与CPU参考最大绝对差{early['max_absolute_loss_difference']:.8g}，通过运行前固定的容差。没有把重放称为历史GPU日志。
- 因此通过预定复用条件，不追加五项TARGET正式训练。

## 三、四组实现检查

- ORACLE-ONLY / SOURCE-ONLY：复用原TargetIQL更新路径；训练中的target buffer槽位装入source。实际128条source/步，没有读取shifted target数据，也没有用其经验统计作归一化。
- 全部999322行的source/cache字段和保留行映射一致，not_done变化17行；source-only两个版本state/action一致。
- TARGET-64：实际64条target/步，没有source抽样。
- ORACLE-W1：实际64+64/步；source_weight_mean全部为1.0；只替换source Q标签，两次Q调用顺序断言通过。Q/V/actor损失均为target均值+source均值。
- 真实shifted评估对象、XML、OTDF归一化、所有标量有限性、35个最终模型/优化器文件、5个V快照均经CPU审计。
- {audit['protected_files_unchanged']}个受保护旧文件哈希未变。原代码、旧实验结果及v3缓存未修改。

## 四、正式队列与耗时

顺序固定为 ORACLE-ONLY seed0–4 → SOURCE-ONLY seed0–4 → TARGET-64 seed0–4 → ORACLE-W1 seed0–4。

共20项、每项500k、总计1000万更新步。前10项完成后生成Q1/Q2中期报告，队列继续运行；最终输出REPORT_v4.md及全部配对统计。Q1–Q4及判据保持冻结协议。

根据四个新组的smoke进度计时，线性外推约 **{estimate:.1f}小时**；v3历史均值外推约51小时。前者包含启动和初始评估后被放大的开销，且不含最后评估尾部，正式策略的回合长度也会变化，因此不是准确完工承诺。建议为正式队列预留约两天的连续运行时间，并在前几项正式完成后更新估计。

本次授权只覆盖GPU冒烟。正式训练需用户确认后创建独立授权记录，不由本报告自动解锁。

## 五、证据入口

- [原始smoke审计](<{(OUT/'smoke_audited.json').as_posix()}>)
- [TARGET复用核验](<{(OUT/'target_reuse_verified.json').as_posix()}>)
- [冻结协议](<{(OUT/'protocol.json').as_posix()}>)
- [实际队列清单](<{(OUT/'training_plan.json').as_posix()}>)
- [方案审查与结论边界](<{(HERE/'REVIEW_AND_PROTOCOL.md').as_posix()}>)
'''
    with open(str(report),'x',encoding='utf-8') as f:f.write(text)
    write(OUT/'smoke_delivery_verified.json',dict(passed=True,report=str(report),report_sha256=sha(report),
        audit_sha256=sha(OUT/'smoke_audited.json'),reuse_sha256=sha(OUT/'target_reuse_verified.json'),
        reporting_script_sha256=sha(__file__),formal_authorized=False,formal_started=False,created=time.time()))
    active=read(OUT/'active_execution.json')
    write(OUT/'launch1/active_before_smoke_delivery.json',active)
    active.update(state='smoke_completed_awaiting_formal_authorization',smoke_report=str(report),smoke_finished=time.time())
    write(OUT/'active_execution.json',active,replace=True)
    write(OUT/'status.json',dict(state='awaiting_formal_authorization',gpu_smoke_runs_complete=5,
        target_reuse_passed=True,formal_authorized=False,formal_runs=0,report=str(report)),replace=True)
    print(str(report),flush=True)


if __name__=='__main__':main()
