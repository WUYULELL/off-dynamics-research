# -*- coding: utf-8 -*-
from report_runtime import *
import torch
def main():
    guard();assert not torch.cuda.is_initialized()
    module=load('_cpu_report_repaired',HERE/'stage1_report_repaired.py');module.main()
    result=base.r.read(REPORT/'stage1/STAGE1_RESULT.json')
    field=base.r.read(REPORT/'CORRECTION_FIELD_AUDIT.json');v=base.r.read(REPORT/'SEED207_VECTOR_AUDIT.json')
    lines=['\n## 汇总脚本修复与配置原文','',
        '原correction是策略损失更正的报告元数据，原脚本期望在训练总config.json读取。历史定义见原v11 prepare.py第26/36行；其原文如下（仅标注历史，不向当前配置补写）：',
        '> '+field['historical_definition'],'',
        '当前训练总config.json没有correction键，但actor_loss保存对应更正说明。修复脚本显式将config.actor_loss映射到报告correction字段，值逐字引用：',
        '> '+field['explicit_mapping']['exact_value'],'',
        '各seed局部config均无correction/actor_loss键，通过scientific_config_sha256指向上述总配置，十项引用哈希均匹配。beta、截断上限、源系数等只按已有字段逐项报告，不自行合成correction文字。',
        '|seed|局部correction|局部actor_loss|引用总配置hash匹配|beta/temp|源actor系数|源项分母|',
        '|---|---|---|---|---:|---:|---:|']
    for c in field['seed_configs']:lines.append('|%d|未单独记录|未单独记录|%s|%s|%s|%s|'%(c['seed'],c['parent_config_matches'],c['temp'],c['actor_source_w'],c['source_denominator']))
    lines+=['','## 原207诊断向量完整性核查','',
        '一行定义为一个训练步的64×7个float32元素。文件预分配500000行；前448506行为有完整JSONL的训练前缀，其后51494行未达到，另行区分。',
        '- 全文件全零行：%d；已记录前缀内全零行：%d。'%(v['zero_rows_full'],v['zero_rows_logged_prefix']),
        '- 前缀非全零行：%d；其中%d行与重跑完全逐位一致，%d行仅部分区域持久化。'%(v['nonzero_rows_logged_prefix'],v['nonzero_rows_bitwise_equal_count'],v['nonzero_rows_unequal_count']),
        '- 104个不完全一致行的所有差异均为旧文件零值位置；旧文件非零值与重跑不一致的元素总数：%d。'%v['mismatching_old_nonzero_cells_total'],
        '- 写入方式：open_memmap(mode=w+)预分配，逐步vectors[step-1]赋值；未按块调用memmap.flush，只在训练循环正常结束后第69行显式flush。JSONL另有每1000步flush，第67行正常结束fsync。',
        '- 结论：中断版本该文件不完整。不能恢复操作系统精确写回时间，但零填充缺口、部分行缺口及非零值逐位一致支持持久化不完整；不把这些零值当成实际训练诊断。',
        '- 确定性结论以V/Q/actor和全部逐步标量诊断JSONL为准：前448506步数值位模式与序列化字节完全一致，首个不一致步为无。不使用回报。',
        '','### 全零行所在步数范围','',str(v['zero_ranges_full']),
        '','### 非全零行所在步数范围','',str(v['nonzero_ranges_logged_prefix']),
        '','104个部分持久化行的逐行差异元素数、完整区间清单及SHA256见SEED207_VECTOR_AUDIT.json。原文件从未改写。',
        '','## 每seed正式结果与配对差值','',
        '|seed|SA-all Last5|Final|减T Last5|减O1 Last5|减raw1 Last5|','|---|---:|---:|---:|---:|---:|']
    r=result['statistics'];g=r['Last5']['groups'];f=r['Final']['groups']
    for i,seed in enumerate(range(201,211)):
        lines.append('|%d|%.5f|%.5f|%+.5f|%+.5f|%+.5f|'%(seed,g['SA-all']['values'][i],f['SA-all']['values'][i],*[g['SA-all']['values'][i]-g[k]['values'][i] for k in ('T','O1','raw1')]))
    for metric in ('Last5','Final'):
        lines+=['','### '+metric+'配对差完整统计','','|参照|差均值|差样本SD|SE|正号|','|---|---:|---:|---:|---:|']
        for ref in ('T','O1','raw1'):
            e=r[metric]['effects']['SA-all'][ref]
            # Existing paired() records use sample_SD for difference dispersion.
            sd=e.get('sample_SD',e.get('SD'))
            if sd is None:
                import numpy as np
                sd=float(np.std(e['differences'],ddof=1))
            lines.append('|%s|%+.5f|%.5f|%.5f|%d/10|'%(ref,e['mean'],sd,e['SE'],e['positive']))
    lines+=['','## 执行出处与停止范围','',
        '原十个逻辑seed中，201–206复用，207从0授权重跑加208–210共四个新完成作业。累计接受500万learner更新；不能把原207中断attempt算成完成seed。原框架new_runs/new_jobs=10指相对60对照的十个方法组记录，本次报告修复新增训练作业数为0。',
        '报告脚本命令：D:\\ANACONDA\\envs\\offdynamics\\python.exe -B oracle_diag/v11_report_fix_20261104/generate_report.py；核验同目录verify_report.py。CUDA_VISIBLE_DEVICES=-1。所有超参数原文见本报告目录config.json（原样副本）；逐seed配置路径与hash见CORRECTION_FIELD_AUDIT.json和训练计划。',
        '只修复报告元数据读取、输出路由和比对解释，原训练/日志/config/audit/失败现场及原冻结文件均保护；报告修复不进行任何优化更新或环境步。SA-top正式、阶段2和任何额外训练未启动。完成后停止。']
    path=REPORT/'stage1/STAGE1_REPORT_ZH.md'
    with open(base.r.winpath(path),'a',encoding='utf-8') as file:file.write('\n'.join(lines)+'\n')
    # Preserve the first repaired analysis receipt; final receipt binds the supplemented report.
    original=REPORT/'stage1/ANALYSIS_COMPLETED.json'
    original.rename(REPORT/'stage1/ANALYSIS_BASE_COMPLETED.json')
    base.r.write(original,dict(passed=True,CPU_only=True,raw_records=70,
        result_sha256=base.r.sha(REPORT/'stage1/STAGE1_RESULT.json'),report_sha256=base.r.sha(path),
        correction_field_audit_sha256=base.r.sha(REPORT/'CORRECTION_FIELD_AUDIT.json'),vector_audit_sha256=base.r.sha(REPORT/'SEED207_VECTOR_AUDIT.json'),time=time.time()))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(report_generated=True,decision=r['Last5']['effects']['SA-all']['decision'],groups={k:dict(mean=x['mean'],SD=x['sample_SD']) for k,x in g.items()},effects=r['Last5']['effects']['SA-all']),ensure_ascii=False))
if __name__=='__main__':main()
