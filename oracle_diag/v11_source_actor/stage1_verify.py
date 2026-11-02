"""Independent raw record/statistic/boundary verification, CPU, no model rollout."""
from common import *
import math
import numpy as np
import torch
from audit import raw_scores

def main():
    guard();assert not torch.cuda.is_initialized()
    a=r.read(STAGE1/'ANALYSIS_COMPLETED.json');result=r.read(STAGE1/'STAGE1_RESULT.json')
    assert a['result_sha256']==r.sha(STAGE1/'STAGE1_RESULT.json')
    assert a['report_sha256']==r.sha(STAGE1/'STAGE1_REPORT_ZH.md')
    for p,h in result['input_sha256'].items():assert r.sha(p)==h,p
    raw={}
    for item in result['records']:
        path=Path(item['out']);score=raw_scores(path,item['seed'],item['every'],item['last'],BC=item['BC'])
        for k in ('Last5','Final'):assert abs(score[k]-item[k])<1e-8
        raw.setdefault(item['label'],[]).append((item['seed'],score))
    assert len(result['records'])==80
    for metric in ('Last5','Final'):
        values={k:[v[metric] for seed,v in sorted(xs)] for k,xs in raw.items()}
        saved=result['statistics'][metric]
        for label,vals in values.items():
            assert len(vals)==10;g=saved['groups'][label]
            mu=sum(vals)/10.;sd=math.sqrt(sum((v-mu)**2 for v in vals)/9.)
            q=sorted(vals);q1=q[2]+.25*(q[3]-q[2])
            assert abs(g['mean']-mu)<1e-8 and abs(g['sample_SD']-sd)<1e-8 and abs(g['Q1']-q1)<1e-8
            assert g['values']==vals and g['worst']==min(vals)
        for variant in ('SA-all','SA-top'):
            x=values[variant];e=saved['effects'][variant]
            for ref in ('T','O1','raw1'):
                diffs=[a-b for a,b in zip(x,values[ref])];mean=sum(diffs)/10.
                se=math.sqrt(sum((v-mean)**2 for v in diffs)/9./10.)
                d=e[ref];assert d['differences']==diffs and abs(d['mean']-mean)<1e-8
                assert abs(d['SE']-se)<1e-8 and d['positive']==sum(v>0 for v in diffs)
            mean=sum(x)/10.-sum(values['BC50'])/10.
            se=math.sqrt(sum((v-np.mean(x))**2 for v in x)/90.+sum((v-np.mean(values['BC50']))**2 for v in values['BC50'])/90.)
            assert abs(e['BC50_Welch']['mean']-mean)<1e-8 and abs(e['BC50_Welch']['SE']-se)<1e-8
            if metric=='Last5':
                b=e['O1'];positive=mean>=5 and mean>2*se and b['mean']>2*b['SE'] and b['positive']>=7
                negative=mean<2 or b['mean']<=0
                assert e['decision']['category']==('成立' if positive else '不成立' if negative else '不确定')
            for ref,diff in e['descriptive_mean_differences'].items():
                assert abs(diff-(np.mean(x)-np.mean(values[ref])))<1e-8
        ds=np.asarray(values['SA-all'])-np.asarray(values['SA-top']);d=saved['SA_all_minus_SA_top']
        assert np.allclose(d['differences'],ds,rtol=0,atol=1e-9) and abs(d['SE']-float(ds.std(ddof=1)/math.sqrt(10)))<1e-8
    for job in formal_jobs():
        p=Path(job['out']);audit=r.read(p/'audit.json');assert audit['passed']
        assert r.sha(p/'model_final.pt')==audit['model_sha256']
        done=r.read(p/'completed.json');assert done['steps']==500000
        for file,h in done['sha256'].items():assert r.sha(file)==h,file
    receipt=protect('Stage1_delivery');assert not torch.cuda.is_initialized()
    r.write(STAGE1/'DELIVERY_VERIFIED.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,
        raw_records=80,new_run_audits=20,statistics_rules_and_descriptive_diffs_recomputed=True,
        original_input_report_model_and_protection_hashes_verified=True,old_files_count=receipt['count'],
        result_sha256=r.sha(STAGE1/'STAGE1_RESULT.json'),report_sha256=r.sha(STAGE1/'STAGE1_REPORT_ZH.md'),time=time.time()))

if __name__=='__main__':main()
