"""Read-only CPU artifact audit; writes only a new final audit receipt."""
import json, hashlib, datetime
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/verified_20261003'
D=OUT/'diagnostics'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):
    if torch.is_tensor(x): assert torch.isfinite(x).all(), 'Nonfinite model tensor'
    elif isinstance(x,dict):
        for v in x.values(): finite(v)
    elif isinstance(x,(list,tuple)):
        for v in x: finite(v)

def main():
    torch.set_num_threads(1)
    plan=read(OUT/'queue_plan.json'); status=read(OUT/'queue_status.json')
    report=read(OUT/'report_data.json'); geometry=read(D/'stage1_stage2a.json')
    assert read(OUT/'final_acceptance.json')['all_passed']
    assert not (OUT/'pipeline_error.json').exists()
    for p,h in plan['protected_sha256'].items(): assert sha(p)==h,p
    rows=[]; models=0; snapshots=0
    for job in status:
        assert job['state'] in ['completed','reused']
        folder=Path(job['out'])
        evidence=next(x for x in report['stage3_score_evidence'] if x['env']==job['env'] and x['method']==job['method'])
        acc=EventAccumulator('\\\\?\\'+str(Path(evidence['file']).resolve()),size_guidance={'scalars':0});acc.Reload()
        tag=next(t for t in acc.Tags()['scalars'] if t.replace(' ','_')=='test/target_normalized_score')
        series={v.step:float(v.value) for v in acc.Scalars(tag)}
        assert max(series)==500000 and all(k in series for k in range(10000,500001,10000))
        assert all(np.isfinite(list(series.values())))
        vals=[series[k] for k in range(460000,500001,10000)]
        assert np.isclose(np.mean(vals),evidence['mean'],atol=1e-10)
        if job['state']=='completed':
            assert job['returncode']==0 and read(folder/'completed.json')['steps']==500000
            assert read(folder/'progress.json')['step']==500000
            dv=read(folder/'data_verified.json'); assert dv['aligned'] and dv['source_n']==4999
            files=list(folder.rglob('model_*'));assert len(files)==7
            for f in files: finite(torch.load(str(f),map_location='cpu'));models+=1
            for step in range(100000,500001,100000):
                finite(torch.load(str(folder/'values'/('V_%d.pt'%step)),map_location='cpu'));snapshots+=1
        else: finite(torch.load(job['value'],map_location='cpu'))
        trainstats={}
        for t in acc.Tags()['scalars']:
            if any(k in t.lower() for k in ['ess','weight_mean','mismatch']):
                tail=[v.value for v in acc.Scalars(t) if 460000<=v.step<=500000]
                if tail:trainstats[t]={'n':len(tail),'mean':float(np.mean(tail))}
        rows.append(dict(env=job['env'],method=job['method'],state=job['state'],evaluation_count=len(series),
                         last5=vals,mean=float(np.mean(vals)),train_tail=trainstats,file=evidence['file']))
    for env,g in geometry.items():
        assert sha(D/(env+'_source_oracle.npz'))==g['oracle_sha256']
        o=np.load(D/(env+'_source_oracle.npz'));a=np.load(D/(env+'_geometry.npz'))
        ids=a['query_indices'];assert len(np.unique(ids))==4999 and np.array_equal(o['indices'],np.arange(4999))
        for key,name in [('pred_a','A_knn_next_mean'),('pred_b','B_displacement'),('pred_c','C_no_shift')]:
            err=np.linalg.norm((a[key]-o['oracle_next'][ids])/o['target_state_std'],axis=1)
            assert np.isclose(err.mean(),g['stage2a'][name]['error']['mean'])
        for kind in ['TARGET','MIX_FB']:
            p=D/(env+'_'+kind+'_bellman.json');r=read(p);a=np.load(p.with_suffix('.npz'))
            assert sha(r['checkpoint'])==r['checkpoint_sha256']
            for key,name in [('knn','knn'),('diff','diff_mean_state'),('diff_per_neighbor','diff_per_neighbor_sensitivity')]:
                m=r['metrics'][name]
                assert np.isclose(spearmanr(a[key],a['oracle']).correlation,m['global_spearman'])
                rho=[spearmanr(a[key][i:i+128],a['oracle'][i:i+128]).correlation for i in range(0,4992,128)]
                assert np.isclose(np.nanmean(rho),m['batch_spearman_mean'])
            for ext in ['png','svg','pdf']:
                fig=OUT/'figures'/(env+'_'+kind+'_bellman.'+ext);assert fig.stat().st_size>1000
    result=dict(checked_at=datetime.datetime.now().astimezone().isoformat(),passed=True,
                protected_files=len(plan['protected_sha256']),new_jobs=14,reused_jobs=1,
                loaded_model_state_files=models,loaded_value_snapshots=snapshots,
                geometry_sets=3,bellman_sets=6,figure_files=18,
                wall_hours=(read(OUT/'pipeline_completed.json')['ended']-plan['created'])/3600,
                boundary_counts={e:len(g['boundary_rows']) for e,g in geometry.items()},jobs=rows,
                scope='CPU artifact/log/hash/calculation audit; simulator and training not rerun; saved artifacts do not prove statistical generalization')
    (OUT/'final_audit_20261005.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='jobs'},ensure_ascii=False))
    print(json.dumps([{'env':r['env'],'method':r['method'],'stats':r['train_tail']} for r in rows if r['method']=='ORACLE'],ensure_ascii=False))

if __name__=='__main__':main()
