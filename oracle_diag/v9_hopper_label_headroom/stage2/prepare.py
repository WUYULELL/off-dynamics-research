# -*- coding: utf-8 -*-
"""Freeze V9 stage2's 20 new runs and 30 R14 comparison records before results."""
from runtime import *
import ast
import difflib
import numpy as np

def main():
    OUT.mkdir(exist_ok=False)
    for name in ('jobs','logs'):(OUT/name).mkdir()
    stage1=read(CACHE_RECEIPT);verified=read(CACHE_STAGE1/'DELIVERY_VERIFIED_STAGE1.json')
    assert stage1['passed'] and stage1['independent_acceptance_passed'] and verified['passed']
    assert sha(stage1['report'])==stage1['report_sha256']==verified['report_sha256']
    assert read(CACHE_STAGE1/'INDEPENDENT_ACCEPTANCE.json')['sample_n']==10000
    for r in stage1['cache_arrays'].values():assert sha(r['path'])==r['sha256']
    original=ROOT/'oracle_diag/results/coverage_raw_20261026_r14'
    end=ROOT/'oracle_diag/results/coverage_raw_20261029_r14_finish1'
    protocol_R14=read(original/'protocol.json');spec=next(t for t in protocol_R14['tasks'] if t['task']=='hopper-morph')
    info=ROOT/'reproductions/OTDF/info.py'
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(info.read_text(encoding='utf-8')).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert spec['score_min']==constants['REF_MIN_SCORE']['hopper-morph'] and spec['score_max']==constants['REF_MAX_SCORE']['hopper-morph']
    protocol=dict(version='V9-stage2-Hopper-label-attribution',tasks=[spec],temperature=10,steps=500000,
        source_weights=[1.,.25],seeds=list(range(201,211)),new_GPU_jobs=20,new_gradient_updates=10000000,
        order='O1 seeds201-210, then O.25 seeds201-210; all20 finish before final attribution',
        independent_variable='replace only source next_state/reward/terminal with accepted stage1 morph labels',
        invariant='source observations/actions/canonical rows, learner, target data, RNG, temperature, batch, weights, schedule, evaluation exactly R14',
        loss='target.mean + w * source.mean for V/Q; no division by1+w; actor_target_only',batch_target=64,batch_source=64,
        source_canonical_rows=999998,target_rows=5000,source_uniform_with_replacement=True,observation_scaling='raw',reward_scale=1.,
        learning_rates=protocol_R14['learning_rates'],evaluation=protocol_R14['evaluation'],
        normalization_reference=str(info),normalization_sha256=sha(info),
        comparison_records=30,analysis_records=50,Last5=[460000,470000,480000,490000,500000],Final_auxiliary=True,
        sample_SD_ddof=1,Q1='linear25percent',paired_SE='sampleSD/sqrt10',positive='strictly greater than0',
        L='O(w) - raw(w), same seed',E='O(w) - T, same seed',
        rule_A='L.mean>2*SE_L AND L.positive>=7 AND E.mean>-2*SE_E',
        rule_A_gain='A AND E.mean>2*SE_E AND E.positive>=7',
        rule_B='NOT(L.mean>2*SE_L) AND E.mean<-2*SE_E',rule_other='uncertain',
        each_w_decided_separately=True,no_changed_thresholds=True,no_bootstrap_report=True,no_stage3_or_4=True,
        no_estimator=True,no_other_experiments=True,stop='after all20 complete, audited, independent raw-data report; wait for user',
        quoted_hours=[48,55],budget_is_estimate_not_guarantee=True)
    write(OUT/'protocol.json',protocol)
    records=[r for r in read(end/'PRECHECK_RESULT.json')['records'] if r['task']=='hopper-morph' and r['label']=='raw']
    assert len(records)==30 and len({(r['seed'],r['w']) for r in records})==30
    reused=[]
    for r in records:
        p=Path(r['out']);a=read(p/'audit.json');d=read(p/'completed.json')
        assert a['passed'] and d['finite'] and d['steps']==500000
        assert sha(d['final_model'])==a['final_model_sha256']==d['final_model_sha256']
        assert sha(p/'completed.json')==a['completed_sha256'] and sha(p/'evaluations.json')==a['evaluations_sha256']
        files={str(f):sha(f) for f in p.rglob('*') if f.is_file() and '__pycache__' not in str(f)}
        reused.append(dict(task='hopper-morph',seed=r['seed'],w=r['w'],label='T' if not r['w'] else 'raw',out=str(p),
            audit=str(p/'audit.json'),audit_sha256=sha(p/'audit.json'),files_sha256=files))
    write(OUT/'reuse_receipts.json',dict(runs=reused,no_R14_run_retrained=True))
    jobs=[]
    for w in (1.,.25):
        for s in range(201,211):
            ident=('hopp_O1_s' if w==1. else 'hopp_O025_s')+str(s)
            job=dict(id=ident,task='hopper-morph',phase='label_attribution',arm='M',seed=s,temp=10.,steps=500000,
                source_w=w,source_lambda=w/(1+w),source_labels='oracle_target_dynamics',out=str(OUT/'runs'/ident))
            jobs.append(job);write(OUT/'jobs'/(ident+'.json'),job)
    write(OUT/'training_plan.json',dict(jobs=jobs,new_GPU_jobs=20,new_updates=10000000,no_followup_training=True))
    protected=dict(read(CACHE_STAGE1/'protected_manifest.json')['sha256'])
    for p in DOC.parent.rglob('*'):
        if p.is_file() and OUT not in p.parents:protected[str(p)]=sha(p)
    # Old v9 stage0/1 code and artifacts are protected; this new stage2 code is separate.
    v9=HERE.parent
    for p in v9.glob('*.py'):protected[str(p)]=sha(p)
    write(OUT/'protected_manifest.json',dict(count=len(protected),sha256=protected))
    oldcode=ROOT/'oracle_diag/coverage_raw_r14_finish1'
    assert sha(HERE/'learner.py')==sha(oldcode/'learner.py')
    same=['seed','configure_cpu','env','target_data','model_config','stream_hash']
    def functions(p):return {n.name:ast.dump(n) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
    oldfunc=functions(oldcode/'runtime.py');newfunc=functions(HERE/'runtime.py')
    assert all(oldfunc[n]==newfunc[n] for n in same)
    def training_loop(p):
        tree=ast.parse(p.read_text(encoding='utf-8'))
        return next(ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    assert training_loop(HERE/'train.py')==training_loop(oldcode/'train.py')
    assert functions(HERE/'train.py')['evaluate']==functions(oldcode/'train.py')['evaluate']
    delta=dict(learner_byte_identical=True,identical_runtime_science_functions=same,training_loop_AST_identical=True,evaluate_AST_identical=True,
        intended_change='source labels only; state/action/indices unchanged',other_changes='new paths and authorization; oracle alignment metadata; oracle-aware CPU audit; Windows long-path atomic IO')
    write(OUT/'engineering_delta.json',delta)
    changes=''.join(difflib.unified_diff((oldcode/'runtime.py').read_text(encoding='utf-8').splitlines(True),(HERE/'runtime.py').read_text(encoding='utf-8').splitlines(True),fromfile='R14/runtime.py',tofile='V9stage2/runtime.py'))
    (OUT/'RUNTIME_DIFF_R14_STAGE2.patch').write_text(changes,encoding='utf-8')
    budget={}
    for w in (.25,1.):
        times=[read(Path(r['out'])/'completed.json')['elapsed']/3600 for r in reused if r['w']==w]
        budget[str(w)]=dict(n=10,mean_hours=float(np.mean(times)),total_hours=float(np.sum(times)))
    write(OUT/'BUDGET_ESTIMATE.json',dict(R14_observed=budget,R14_twenty_mixed_total_hours=sum(v['total_hours'] for v in budget.values()),quoted_wall_hours=[48,55],
        assumptions='continuous single GPU, same R14 compute, variable evaluation lengths and no competing GPU load',includes='20 new500k runs, per-run CPU audit, final raw-input analysis'))
    write(OUT/'stage_authorization.json',dict(scope='V9_stage2_hopper_label_attribution_20_only',protocol_sha256=sha(OUT/'protocol.json'),
        new_GPU_jobs=20,explicit_user_request='2026-10-30 continue subsequent stage after stage1 delivery',estimator_development_authorized=False,
        stage3_or_4_authorized=False,time=time.time()))
    user=Path(read(DOC.parent/'config.json')['source_instruction_path'])
    assert sha(user)==read(DOC.parent/'config.json')['source_instruction_sha256']
    config=dict(protocol=protocol,oracle_receipt=str(CACHE_RECEIPT),oracle_receipt_sha256=sha(CACHE_RECEIPT),
        learner_common=model_config(10,500000,201,'hopper-morph',1.),stage1_independent_acceptance_sha256=sha(CACHE_STAGE1/'INDEPENDENT_ACCEPTANCE.json'),
        seeds=protocol['seeds'],all_hyperparameters_in_protocol=True,code_science_parity=delta)
    write(OUT/'config.json',config)
    deps=[user,info,Path(spec['xml']),Path(spec['source_dataset']),Path(spec['target_dataset']),CACHE_RECEIPT,
        CACHE_STAGE1/'DELIVERY_VERIFIED_STAGE1.json',CACHE_STAGE1/'INDEPENDENT_ACCEPTANCE.json',CACHE_STAGE1/'cache/raw_indices.npy',
        OUT/'protocol.json',OUT/'config.json',OUT/'training_plan.json',OUT/'reuse_receipts.json',OUT/'stage_authorization.json']
    deps += [Path(v['path']) for v in stage1['cache_arrays'].values()]
    deps += [OUT/'jobs'/(j['id']+'.json') for j in jobs]
    write(OUT/'dependency_hashes.json',dict(sha256={str(p):sha(p) for p in deps}))
    code=list(HERE.glob('*.py'))+[ROOT/'algo/offline_offline'/n for n in ('iql.py','iql_dyn.py','iql_unc.py')]
    write(OUT/'code_frozen.json',dict(sha256={str(p):sha(p) for p in code}))
    write(OUT/'PRE_RESULT_COMMITMENT.json',dict(time=time.time(),before_new_long_run=True,protocol_sha256=sha(OUT/'protocol.json'),
        training_plan_sha256=sha(OUT/'training_plan.json'),code_frozen_sha256=sha(OUT/'code_frozen.json'),known_R14_results_seen=True,
        no_new_stage2_results=True,no_threshold_change=True,source_instruction_sha256=sha(user)))
    print(json.dumps(dict(prepared=True,jobs=20,reuse=30,protected=len(protected),budget_hours=[48,55])))

if __name__=='__main__':main()
