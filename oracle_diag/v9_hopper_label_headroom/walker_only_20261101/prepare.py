from common import *
import ast,statistics
def main():
    if OUT.exists():
        assert not (OUT/'launch1.json').exists() and not (OUT/'FROZEN.json').exists()
        assert r.read(OUT/'config.json')['scope']=='Walker15_only_then_stop'
    else:r.fresh(OUT)
    (OUT/'logs').mkdir(exist_ok=True)
    hopper=ROOT/'outputs/v9_hopper_label_headroom_20261101/hopper_stage3_correction'
    hv=r.read(hopper/'DELIVERY_VERIFIED.json');assert hv['passed'] and r.read(hopper/'HOPPER_ONLY_COMPLETED.json')['passed']
    assert r.sha(hopper/'HOPPER_REPORT_ZH.md')==hv['report_sha256']
    trajectory=r.read(hopper/'config.json')['trajectory_selections']['walker2d-morph']
    assert trajectory['n_complete']==7
    assert trajectory['variants']['BC50']['segments']==[6,4,7,3] and trajectory['variants']['BC25']['segments']==[6,4]
    original=r.read(OLD/'selections.json')['walker2d-morph']
    for key in ('BC_all','BC50','BC25'):assert trajectory['variants'][key]['rows']==original['variants'][key]['rows']
    assert [len(trajectory['variants'][k]['rows']) for k in ('BC_all','BC50','BC25')]==[5000,3668,2000]
    plan=jobs();assert not list((OLD/'runs').glob('walk_*'))
    timings=[r.read(p)['elapsed'] for p in (OLD/'runs').glob('hopp_*/completed.json')];assert len(timings)==30
    config=dict(scope='Walker15_only_then_stop',task='walker2d-morph',seeds=list(range(301,306)),variants=['BC_all','BC50','BC25'],
        trajectory_selection=trajectory,steps=200000,cosine_T_max=200000,batch=64,beta=0,advantage_weight=1,actor_lr=.0003,
        actor='same frozen R14 IQL_DYN Policy as Hopper',loss='mean_batch(mean_action(MSE))',Q_V_constructed=False,Q_V_optimized=False,source_rows=0,
        eval_every=5000,episodes=10,Last5=[180000,185000,190000,195000,200000],Final=200000,
        eval_seeds='500000+10*(trainingseed-301)+episode0..9',save_restore_training_RNG=True,
        T_reference='R13 Walker T seeds201–205, unchanged500k/temp10',Welch_unpaired_to_T=True,
        descriptive_references=['R13_U','R13_G5','R13_R5','R14_w025'],descriptive_only_no_new_gates=True,
        total_actor_updates=3000000,stage4_authorized=False,T_tuning_authorized=False,
        expected_total_hours=[3,4],Hopper_observed_mean_minutes=statistics.mean(timings)/60,
        Hopper15_equivalent_hours=statistics.mean(timings)*15/3600,estimate_reported_to_user_before_launch=True,
        selection_bias_C_disclosed=True,Hopper_completed_and_verified=True,time=time.time())
    if (OUT/'config.json').exists():
        previous=r.read(OUT/'config.json');config['time']=previous['time'];assert config==previous
        assert r.read(OUT/'training_plan.json')==dict(jobs=plan)
    else:
        r.write(OUT/'config.json',config);r.write(OUT/'training_plan.json',dict(jobs=plan))
    r13path=ROOT/'oracle_diag/results/coverage_oracle_20261024_r13_finish1/CONFIRMATION_RESULT.json'
    r13=r.read(r13path);references=[]
    prior_reuse=r.read(r13path.parent/'reuse_receipts.json')['runs']
    reuse_by_path={v['job']['out']:v for v in prior_reuse}
    for arm,label in [('T','T'),('U','R13_U'),('G','R13_G5'),('R','R13_R5')]:
        values=r13['runs'][arm];assert len(values)==5 and [v['job']['seed'] for v in values]==list(range(201,206))
        for record in values:
            p=Path(record['job']['out']);audit_path=p/'audit.json';extra={}
            if not audit_path.exists():
                saved=reuse_by_path[str(p)];audit_path=Path(saved['audit']);assert r.sha(audit_path)==saved['audit_sha256']
                derived=Path(record['derived_point_receipt']);assert r.sha(derived)==record['derived_point_sha256']
                repair=Path(saved['audit_repair_receipt']);assert r.read(repair)['passed'] and r.read(repair)['repaired_audit_sha256']==r.sha(audit_path)
                extra=dict(derived_receipt=str(derived),derived_receipt_sha256=record['derived_point_sha256'],audit_repair_receipt=str(repair),
                    TB_exception='49 original TB points;100k derived from pre-pause sealed raw episodes;Last5/Final original TB complete')
            audit=r.read(audit_path);done=r.read(p/'completed.json')
            assert record['passed'] and audit['passed'] and done['steps']==500000 and done['finite']
            assert r.sha(done['final_model'])==record['final_model_sha256']==done['final_model_sha256']
            assert r.sha(p/'evaluations.json')==record['evaluations_sha256']
            references.append(dict(label=label,seed=record['job']['seed'],task='walker2d-morph',out=str(p),
                provenance=str(r13path),audit=str(audit_path),audit_sha256=r.sha(audit_path),final_model_sha256=done['final_model_sha256'],**extra))
    r14path=ROOT/'oracle_diag/results/coverage_raw_20261029_r14_finish1/PRECHECK_RESULT.json';r14=r.read(r14path)
    values=[v for v in r14['records'] if v['task']=='walker2d-morph' and v['w']==.25];assert len(values)==5
    for v in values:
        p=Path(v['out']);assert r.read(p/'audit.json')['passed'] and r.sha(p/'audit.json')==v['audit_sha256']
        done=r.read(p/'completed.json');assert r.sha(done['final_model'])==done['final_model_sha256']
        references.append(dict(label='R14_w025',seed=v['seed'],task=v['task'],out=str(p),provenance=str(r14path),audit=str(p/'audit.json'),audit_sha256=v['audit_sha256'],final_model_sha256=done['final_model_sha256']))
    assert len(references)==25;r.write(OUT/'reference_receipts.json',dict(records=references,R13_R_recovery_provenance=r13['recovery']))
    r.write(OUT/'BUDGET_ESTIMATE.json',dict(Hopper_completed_runs=30,Hopper_mean_minutes=statistics.mean(timings)/60,
        direct15_run_extrapolation_hours=statistics.mean(timings)*15/3600,quoted_Walker_total_hours=[3,4],
        assumptions='same single GPU, no competing load; reserve for Walker episode length, per-run audits and final verification',
        quoted_before_any_Walker_GPU_training=True,no_new_timing_probe=True,time=time.time()))
    r.write(OUT/'authorization.json',dict(explicit_user_instruction='Confirm v9 stage3 Walker only, seeds301–305, BC-all/BC50(6,4,7,3)/BC25(6,4), same Hopper settings; report budget before launch; compare descriptively U/G5/R5/R14w.25; stop, no stage4 or T tuning',new_GPU_jobs=15,no_additional_training=True,time=time.time()))
    r.write(OUT/'PRE_RESULT_COMMITMENT.json',dict(time=time.time(),before_any_Walker_BC_results=True,Hopper_results_seen=True,
        protocol_sha256=r.sha(OUT/'config.json'),plan_sha256=r.sha(OUT/'training_plan.json'),references_sha256=r.sha(OUT/'reference_receipts.json'),
        V9_rules_unchanged=True,C_selection='largest Last5 BC mean; exact ties fixed BC_all/BC50/BC25 order, disclose ties',
        positive='H>=10 and H>2SE_H',insufficient='H<5',uncertain='5<=H<10',uncovered='H>=10 and H<=2SE_H: stop and report rule gap',
        weak_T='BC_all-T > 2WelchSE',all_other_reference_comparisons_descriptive_only=True))
    # Same actor/data/evaluation code as the accepted Hopper run; no new real-data optimization or environment step.
    r.configure_cpu()
    from learner import Actor,Sampler
    import torch
    expectations=r.read(OLD/'INITIALIZATION_EXPECTATIONS.json')['sha256'];arrays=r.target_data('walker2d-morph')
    for j in plan:
        assert Actor(j,'cpu').digest()==expectations['walker2d-morph|'+str(j['seed'])]
        sampler=Sampler(arrays,j,'cpu');sampler.sample();sampler.sample();assert sampler.hash.hexdigest()==r.sample_hash(j,2)
    for ref in references:
        p=Path(ref['out']);v=raw_score_record(p,ref,10000,500000);a=r.read(ref['audit'])
        assert abs(v['Last5']-a['Last5'])<1e-8 and abs(v['Final']-a['Final'])<1e-8
    assert not torch.cuda.is_initialized()
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    r.write(OUT/'CPU_ACCEPTANCE.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,new_optimizer_updates=0,new_environment_steps=0,
        original8_synthetic_actor_updates_acceptance_reused=True,Walker_part_original_synthetic_updates=4,
        all15_initializations_and_sample_prefixes_match_Hopper_implementation=True,all25_references_raw_TB_episode_recomputed=True,
        selected_indices_match_exactly=True,all_modules_syntax_passed=True,time=time.time()))
    protected=dict(r.read(OLD/'protected_manifest.json')['sha256'])
    for base in [OLD,hopper,ROOT/'oracle_diag/v9_hopper_label_headroom/stage3',ROOT/'oracle_diag/v9_hopper_label_headroom/hopper_only_20261101']:
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=r.sha(p)
    r.write(OUT/'protected_manifest.json',dict(sha256=protected,time=time.time()))
    frozen={str(p):r.sha(p) for p in HERE.glob('*.py')}
    for name in ('config.json','training_plan.json','reference_receipts.json','BUDGET_ESTIMATE.json','authorization.json','PRE_RESULT_COMMITMENT.json','CPU_ACCEPTANCE.json'):
        frozen[str(OUT/name)]=r.sha(OUT/name)
    frozen[str(r13path)]=r.sha(r13path);frozen[str(r14path)]=r.sha(r14path)
    for ref in references:
        p=Path(ref['out'])
        for name in ('completed.json','evaluations.json'):frozen[str(p/name)]=r.sha(p/name)
        frozen[ref['audit']]=r.sha(ref['audit'])
        for name in ('derived_receipt','audit_repair_receipt'):
            if name in ref:frozen[ref[name]]=r.sha(ref[name])
    r.write(OUT/'FROZEN.json',dict(sha256=frozen,time=time.time()))
    guard();print(json.dumps(dict(passed=True,Walker_jobs=15,reused_reference_runs=25,estimated_hours=[3,4],real_optimizer_updates=0,environment_steps=0)))
if __name__=='__main__':main()
