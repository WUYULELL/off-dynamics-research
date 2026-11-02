"""Generate an immutable Chinese interim/final report only from audited runs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support_v4 import *
from audit_v4 import paired, baselines
import argparse
import statistics
import math


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('interim','final'),required=True);a=p.parse_args()
    is_final=a.kind=='final'
    source=OUT/('formal_audited.json' if is_final else 'interim_audited.json')
    audit=read(source);assert audit['passed']
    assert read(OUT/'target_reuse_verified.json')['passed']
    rows=audit['runs']; assert len(rows)==(20 if is_final else 10)
    old=baselines(); values={(r['method'],r['seed']):r['last5_mean'] for r in old+rows}
    protocol=read(OUT/'protocol.json'); comparisons={}
    for question,(left,right) in protocol['main_comparisons'].items():
        if not is_final and question not in ('Q1','Q2'):continue
        comparisons[question]=dict(left=left,right=right,**paired([values[left,s]-values[right,s] for s in range(5)]))
    descriptions={}
    for left,right in protocol['descriptive_comparisons']:
        if all((left,s) in values and (right,s) in values for s in range(5)):
            descriptions[left+' − '+right]=paired([values[left,s]-values[right,s] for s in range(5)])
    # Independent NumPy arithmetic check, separate from the paired helper.
    import numpy as np
    for row in list(comparisons.values())+list(descriptions.values()):
        x=np.asarray(row['values'],dtype=np.float64)
        assert np.isclose(row['mean'],x.mean(),rtol=0,atol=1e-12)
        assert np.isclose(row['sample_sd'],x.std(ddof=1),rtol=0,atol=1e-12)
        assert np.isclose(row['se'],x.std(ddof=1)/np.sqrt(5),rtol=0,atol=1e-12)
    methods=list(METHODS) if is_final else list(SINGLE_SOURCE)
    title='Oracle v4 归因实验最终报告' if is_final else 'Oracle v4 中期报告：前两组已完成'
    lines=['# '+title,'',
        '本报告从经核验的原始 TensorBoard 事件计算 Last-5。'+('四组共20项正式训练完成。' if is_final else 'ORACLE-ONLY、SOURCE-ONLY各五个seed完成；后两组继续排队，不据中期结果变更判据。'),'',
        '## 一、直接结果','']
    result_rows=[]
    for q,row in comparisons.items():
        result_rows.append([q,row['left']+' − '+row['right'],f"{row['mean']:+.4f}",f"{row['se']:.4f}",
                            str(row['positive'])+'/5','通过预设正向门槛' if row['passed'] else '未通过；未观察到稳定正向差异'])
    lines += [table(['比较','定义','平均差','SE','正号数','结论'],result_rows),'',
        '门槛在运行前固定：配对均值 > 2×SE，且至少4/5个seed为正。这是工程决策门槛，不是严格95%置信区间；未通过也不等于证明无效。','',
        '## 二、实现、抽样与复用核验','',
        '- ORACLE-ONLY / SOURCE-ONLY 直接使用冻结的 v3 TargetIQL，把全部source装入target buffer槽位，每步128条；state/action不变，仅标签来源不同。三项损失均为batch均值，无source折扣。',
        '- 这两组不读取shifted target文件，不使用其经验归一化统计。评分仅使用事先固定的OTDF参考常数。',
        '- TARGET-64 每步实际抽64条target；ORACLE-W1每步64+64，三项损失均为target均值+1.0×source均值，仅source Q调用替换oracle标签。每步计数和两次Q调用顺序均有断言。',
        '- 全部999322条source/cache行在CPU上核对；not_done变化17行；缓存使用v3验收路径与SHA256。',
        '- TARGET复用：v4入口seed0/5000步与历史smoke的初始化、首批索引、有效配置一致，最终模型及评估满足预先冻结容差。v3未保存早期损失；额外CPU重放与v4短步逐位一致，并与新GPU早期损失在预定容差内一致。不是补造历史日志。',
        '- 原始train.py通过runpy运行；环境、显式next_state loader、OTDF归一化、seed和优化器配置沿用v3。没有source/target方向交换。','',
        '## 三、逐seed Last-5','']
    lines += [table(['seed','TARGET（复用）']+methods,
        [[s]+[f'{values[m,s]:.4f}' for m in ['TARGET']+methods] for s in range(5)]),'']
    arm_rows=[]
    for method in ['TARGET']+methods:
        x=[values[method,s] for s in range(5)]
        arm_rows.append([method,f'{statistics.mean(x):.4f}',f'{statistics.stdev(x):.4f}',f'{statistics.stdev(x)/math.sqrt(5):.4f}'])
    lines += [table(['方法','均值','样本SD','SE'],arm_rows),'','## 四、完整配对统计','']
    for label,records in [('预设Q比较',comparisons),('描述性比较',descriptions)]:
        lines += ['### '+label,'',table(['比较','seed0..4差值','均值','样本SD','SE','正号数'],
            [[name,', '.join(f'{x:+.4f}' for x in row['values']),f"{row['mean']:+.4f}",
              f"{row['sample_sd']:.4f}",f"{row['se']:.4f}",str(row['positive'])+'/5'] for name,row in records.items()]),'']
    q1=comparisons['Q1']['passed'];q2=comparisons['Q2']['passed']
    lines += ['## 五、归因解释及行动判断','']
    if q1:
        lines.append('Q1通过：这套oracle标注的source-only训练方案优于本轮TARGET。它支持继续研究这套干预，但没有单独识别“状态覆盖”的作用；source动作分布、数据量、训练路径也不同。')
        if q2:lines.append('Q2也通过：在相同source-only更新中，更换一步标签带来稳定正向差异，支持把近似这些标签作为后续候选方向。')
        else:lines.append('Q2未通过：尚不能把Q1收益归功于标签修正，暂不足以支持扩大标签修正器研究。')
    elif q2:
        lines.append('Q1未通过、Q2通过：修正标签改善了source-only，但未证明它优于TARGET。状态分布不匹配只是候选解释，也可能涉及学得的V、策略约束、优化或样本波动；不能给出动力学修正的理论性能上界。')
    else:
        lines.append('Q1、Q2均未通过：当前任务、当前数据和当前IQL路径下，没有稳定正向证据支持扩大这条标签修正方案。不能据此认定覆盖与标签都不是瓶颈，更不能推出所有kNN/OT/Bellman方法无效。')
    if is_final:
        q3=comparisons['Q3']['passed'];q4=comparisons['Q4']['passed']
        lines.append('Q3'+('通过：target-only从128降到64在本轮造成稳定退化；这为v3差距提供了一个候选解释，但混合训练可能有交互作用，不能将该差值直接当作v3差距中batch因素的因果份额。' if q3 else '未通过：未观察到target-only中batch128相对64的稳定正向差异；不等于batch绝无影响。'))
        lines.append('Q4'+('通过：权重1.0的oracle混合方案优于TARGET。与v3权重0.2的描述性对照一起，可评估权重修改是否值得进一步验证；Q4本身不能证明“问题只在权重”。' if q4 else '未通过：未观察到权重1.0的oracle混合相对TARGET的稳定收益；不能单凭它证明target数据干扰或混合必然有害。'))
        lines.append('Q4改变了source对Q、V、actor三项损失的系数，也改变总损失尺度；本轮没有把这些路径进一步分开。')
    lines += ['', '## 六、标签变化诊断','']
    ready=read(OUT/'cpu_ready.json');data=read(Path(ready['cpu_attempt'])/'data_verified.json')
    diagnostic_rows=[]
    for label,key in [('next_state逐行L2距离','next_state_row_l2'),('next_state逐行最大绝对差','next_state_row_max_abs'),('reward绝对差','reward_abs')]:
        d=data[key];diagnostic_rows.append([label,f"{d['mean']:.6g}",f"{d['p50']:.6g}",f"{d['p95']:.6g}",f"{d['p100']:.6g}"])
    lines += [table(['量','均值','中位数','P95','最大值'],diagnostic_rows),'',
        '这是正常source记录与shifted oracle标签的差异，不是回放重建误差；状态坐标单位不同，未把L2解释为任务价值。','',
        '## 七、局限与证据范围','',
        '- 单任务、五个优化seed，同一批target约8个轨迹段；没有覆盖独立数据采集的不确定性。',
        '- oracle使用历史恢复的隐藏物理状态；缓存的事实回放验收支持实现可信度，并非反事实条件转移的独立真值证明。',
        '- V是学得函数。一步转移正确不保证长期估值或最终策略最优。',
        '- 复用v3 TARGET与v4运行时间不同。smoke复用检查降低管线漂移风险，不能证明所有运行时条件完全相同。',
        '- ORACLE-ONLY对比TARGET同时改变状态、动作覆盖及数据量；Q3属于target-only干预，不能无条件外推到混合流程。',
        '- 四个Q比较共用数据和基线，门槛不是经多重比较校正的显著性检验。',
        '- 不与OTDF/TABB论文数字并列排名，不把oracle诊断当作无需模拟器的方法。','',
        '## 附录A：每个checkpoint原始分数','',
        table(['方法','seed','460k','470k','480k','490k','500k'],
            [[r['method'],r['seed']]+[f'{x:.8f}' for x in r['last5']] for r in old+rows if r['method']=='TARGET' or r in rows]),'',
        '## 附录B：核验来源','']
    for label,path in [('冻结协议',OUT/'protocol.json'),('训练清单',OUT/'training_plan.json'),('本轮审计',source),
                       ('TARGET复用验收',OUT/'target_reuse_verified.json'),('CPU数据和公式检查',Path(ready['cpu_attempt'])/'completed.json'),
                       ('v3原结果',V3_CODE/'DELIVERY_REPORT_v3_20261011.md')]:
        assert path.exists();lines.append(f'- [{label}](<{path.as_posix()}>)')
    text='\n'.join(lines)+'\n'
    assert '\ufffd' not in text
    report=HERE/('REPORT_v4.md' if is_final else 'INTERIM_v4.md')
    with open(str(report),'x',encoding='utf-8') as f:f.write(text)
    write(OUT/(a.kind+'_report_verified.json'),dict(passed=True,report=str(report),report_sha256=sha(report),
        audit_sha256=sha(source),comparisons=comparisons,descriptive=descriptions,created=time.time(),
        recomputed_from_raw_baselines=True,independent_numpy_statistics=True))
    print(str(report),flush=True)


if __name__=='__main__':main()
