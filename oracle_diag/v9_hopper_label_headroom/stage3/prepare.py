from runtime import *
import numpy as np
import h5py

def main():
    fresh(OUT);(OUT/'jobs').mkdir();(OUT/'logs').mkdir()
    prior=ROOT/'outputs/v9_hopper_label_headroom_20261030'
    raw_protocol=read(ROOT/'oracle_diag/results/coverage_raw_20261026_r14/protocol.json')
    tasks=raw_protocol['tasks'];ledger=read(prior/'TARGET_TRAJECTORIES.json')
    write(OUT/'protocol.json',dict(stage=3,tasks=tasks,order=['hopper-morph','walker2d-morph'],
        variants=['BC_all','BC50','BC25'],steps=200000,evaluate_every=5000,evaluation_episodes=10,Last5=[180000,185000,190000,195000,200000],
        actor='exact R14 IQL_DYN Policy from iql_unc',hidden=256,actor_lr=.0003,batch=64,beta=0,advantage_weight=1,
        policy_loss='Policy.bc_loss.mean(action).mean(batch), exact R14 IQL_DYN beta0 reduction order',no_Q_V_training=True,no_source=True,cosine_T_max=200000,
        headroom_rules=dict(positive='H>=10 and H>2SE_H',insufficient='H<5',uncertain='5<=H<10',
            uncovered='H>=10 and H<=2SE_H: stop interpretation and ask user',T_weak='BC_all-T > 2*Welch_SE'),
        C_selection='max Last5 mean; exact ties use fixed BC_all,BC50,BC25 order and disclose all tied groups',
        unpaired_comparison_to_T=True,no_bootstrap=True,stage4_authorized=False))
    selections={}
    for spec in tasks:
        task=spec['task'];old=ledger[task];assert sha(spec['target_dataset'])==spec['target_dataset_sha256']==old['sha256']
        with h5py.File(spec['target_dataset'],'r') as f:
            rewards=np.asarray(f['rewards'],dtype=np.float64).reshape(-1);obs=np.asarray(f['observations']);nxt=np.asarray(f['next_observations']);term=np.asarray(f['terminals']).reshape(-1)
        segments=old['reconstructed_segments'];assert len(segments)==8 and sum(s['length'] for s in segments)==5000
        for seg in segments:
            a,b=seg['start_row'],seg['end_row'];assert seg['length']==b-a+1
            assert abs(rewards[a:b+1].sum()-seg['raw_return'])<1e-8
            if seg['end_reason']=='terminal':assert term[b]==1
            if seg['end_reason'].startswith('inferred'):assert np.max(np.abs(nxt[b]-obs[b+1]))>1e-4
        ranked=sorted(segments,key=lambda s:(-s['raw_return'],s['segment']))
        complete=sorted([s for s in segments if s['complete_boundary']],key=lambda s:(-s['raw_return'],s['segment']))
        variants={'BC_all':dict(rows=list(range(5000)),segments=[s['segment'] for s in segments])}
        for name,fraction in [('BC50',.5),('BC25',.25)]:
            k=math.ceil(len(segments)*fraction);selected=ranked[:k]
            assert [s['segment'] for s in selected]==[s['segment'] for s in complete[:math.ceil(len(complete)*fraction)]]
            assert all(s['complete_boundary'] for s in selected)
            rows=sorted(i for s in selected for i in range(s['start_row'],s['end_row']+1))
            variants[name]=dict(rows=rows,segments=[s['segment'] for s in selected],fraction=fraction,k=k)
        selections[task]=dict(n=8,complete_n=7,EOF_partial_return_disclosed=True,BC_all_uses_all5000=True,
            selected_rows_identical_for_n7_or8=True,no_BC10_because_n_lt20=True,segments=segments,variants=variants)
    write(OUT/'selections.json',selections)
    jobs=[]
    for task,short,end in [('hopper-morph','hopp',311),('walker2d-morph','walk',306)]:
        for variant in ('BC_all','BC50','BC25'):
            for s in range(301,end):
                name='%s_%s_s%d'%(short,variant,s)
                j=dict(id=name,task=task,variant=variant,seed=s,steps=200000,beta=0,batch=64,out=str(OUT/'runs'/name))
                jobs.append(j);write(OUT/'jobs'/(name+'.json'),j)
    assert len(jobs)==45;write(OUT/'training_plan.json',dict(jobs=jobs,Hopper_first30=True,Walker_follows15=True,total_actor_updates=9000000))
    old=read(ROOT/'oracle_diag/results/coverage_raw_20261029_r14_finish1/PRECHECK_RESULT.json')
    refs=[{k:v[k] for k in ('task','seed','out','audit','files_sha256')} for v in old['records'] if v['w']==0.]
    assert len(refs)==15
    for item in refs:
        for p,h in item['files_sha256'].items():assert sha(p)==h
        assert read(item['audit'])['passed']
    write(OUT/'T_reuse.json',dict(records=refs,Hopper_R14_n10=True,Walker_R13_n5=True))
    write(OUT/'authorization.json',dict(new_jobs=45,order=['hopper-morph','walker2d-morph'],
        explicit_user_instruction='2026-10-31 O.25继续跑完，之后先Hopper BC/%BC，Walker放在后面；v9不改',
        no_interruption_or_changes_to_stage2=True,stage4_authorized=False,estimator_authorized=False,time=time.time()))
    frozen={str(OUT/n):sha(OUT/n) for n in ('protocol.json','selections.json','training_plan.json','T_reuse.json','authorization.json')}
    for p in (OUT/'jobs').iterdir():frozen[str(p)]=sha(p)
    for p in (prior/'sources/V9_INSTRUCTIONS.md',prior/'TARGET_TRAJECTORIES.json',ROOT/'reproductions/OTDF/info.py'):
        frozen[str(p)]=sha(p)
    for task in tasks:
        for key in ('xml','target_dataset'):frozen[task[key]]=sha(task[key])
    write(OUT/'input_frozen.json',dict(sha256=frozen))
    write(OUT/'PRE_RESULT_COMMITMENT.json',dict(time=time.time(),known_O1_result_seen=True,no_BC_results_seen=True,
        stage2_still_running=True,protocol_sha256=sha(OUT/'protocol.json'),selections_sha256=sha(OUT/'selections.json'),
        plan_sha256=sha(OUT/'training_plan.json'),thresholds_from_original_V9=True))
    print(json.dumps(dict(jobs=len(jobs),selections={task:{k:len(v['rows']) for k,v in d['variants'].items()} for task,d in selections.items()})))

if __name__=='__main__':main()
