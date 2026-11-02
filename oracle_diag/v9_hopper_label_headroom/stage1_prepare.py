"""Prepare a new CPU-only Hopper cache stage without simulator steps."""
from stage1_runtime import *
import ast
import numpy as np

def main():
    OUT.mkdir(exist_ok=False)
    for n in ('logs','cache','acceptance'):(OUT/n).mkdir()
    stage0=read(BASE/'STAGE0_COMPLETED.json')
    assert stage0['passed'] and read(BASE/'DELIVERY_VERIFIED_STAGE0.json')['passed']
    user=Path(read(BASE/'config.json')['source_instruction_path'])
    assert sha(user)==read(BASE/'config.json')['source_instruction_sha256']
    raw=load_raw();ids=np.flatnonzero(~raw['timeouts'][:-1].astype(bool))
    assert len(ids)==999998
    np.save(wp(OUT/'cache/raw_indices.npy'),ids)
    sample=np.random.RandomState(20261030).choice(len(ids),10000,replace=False)
    np.save(wp(OUT/'acceptance/sample_buffer_indices.npy'),sample)
    np.save(wp(OUT/'acceptance/sample_raw_indices.npy'),ids[sample])
    projection=np.concatenate((raw['infos/qpos'][:,1:],np.clip(raw['infos/qvel'],-10,10)),axis=1)
    assert np.array_equal(projection,raw['observations'])
    info=ROOT/'reproductions/OTDF/info.py'
    const={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(info.read_text(encoding='utf-8')).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    r14=read(ROOT/'oracle_diag/results/coverage_raw_20261026_r14/protocol.json')
    spec=next(s for s in r14['tasks'] if s['task']=='hopper-morph')
    assert spec['source_dataset_sha256']==sha(SOURCE) and spec['xml_sha256']==sha(XML)
    protected=dict(read(BASE/'OLD_FILE_PROTECTION_BEFORE.json')['sha256'])
    # Preserve every pre-existing v9 stage0 output and all three stage0 programs.
    for p in BASE.rglob('*'):
        if p.is_file() and OUT not in p.parents:protected[str(p)]=sha(p)
    for p in HERE.glob('stage0_*.py'):protected[str(p)]=sha(p)
    write(OUT/'protected_manifest.json',dict(count=len(protected),sha256=protected))
    write(OUT/'authorization.json',dict(stage=1,explicit_user_request='2026-10-30: start subsequent stage after stage0 delivery',
        scope='normal Hopper source -> Hopper-morph oracle cache and independent>=10000 replay acceptance only',
        RL_training_authorized=False,stage2_3_4_authorized=False,stop_after_stage1=True,time=time.time()))
    config=dict(stage=1,CPU_only=True,task='hopper-morph',source=str(SOURCE),source_sha256=sha(SOURCE),
        raw_rows=1000000,canonical_rows=999998,canonical_rule='exclude source timeout row and last raw row, exactly as R14',
        target_xml=str(XML),target_xml_sha256=sha(XML),source_environment="gym.make('hopper-medium-v2')",
        target_environment='gym HopperEnv v3 with exact OTDF XML; TimeLimit1000; frame_skip4; timestep.002',
        label_fields=['next_state','reward','terminal','not_done'],cache_dtype='next_state/reward/not_done float32, terminal uint8',
        source_labels_used_for_factual_acceptance='raw explicit next_observations, rewards, terminals; no adjacent-obs rewrite',
        no_policy_fitting=True,no_GPU=True,training_updates=0,normalization_reference=str(info),normalization_sha256=sha(info),
        min_score=const['REF_MIN_SCORE']['hopper-morph'],max_score=const['REF_MAX_SCORE']['hopper-morph'],reward_scaling='none',
        state_recovery='raw qpos, non-clipped qvel; propagate verified chronological source transitions inside episodes for clipped velocity; qpos[0]=0 translation symmetry',
        fallback_history_steps=[1,2,4,8,16,32,64],history_consistency_tolerance=1e-4,
        factual_state_Linf_tolerance=1e-3,factual_reward_abs_tolerance=1e-4,factual_terminal_match='exact',
        restored_observation_Linf_tolerance=1e-4,target_replay_state_Linf_tolerance=1e-3,target_replay_reward_abs_tolerance=1e-4,
        target_terminal_match='exact',no_failed_canonical_rows_removed=True,no_tolerance_relaxation=True,
        independent_replay_n=10000,independent_sample_seed=20261030,sample_without_replacement=True,
        independent_recovery='fresh source and target environments; recompute physical state from raw nearest unclipped in-episode anchor; never consume generated qpos/qvel as input',
        reset_convention='sim.reset -> set_state -> qacc_warmstart=0 -> one physical environment step, same as accepted Walker cache',
        observed_clipped_rows=171833,maximum_clipped_run=15,estimated_CPU_hours=[.5,2.],estimate_not_measured_Hopper=True,
        stop_on_any_acceptance_failure=True,followup_training=False)
    write(OUT/'config.json',config)
    refs=[SOURCE,XML,info,user,BASE/'STAGE0_COMPLETED.json',BASE/'DELIVERY_VERIFIED_STAGE0.json',
        ROOT/'oracle_diag/v3/cache_source_v3.py',ROOT/'oracle_diag/v3/recover_source_v3.py',ROOT/'oracle_diag/v3/stage_a_v3.py',
        ROOT/'oracle_diag/results/v3_20261008/source_oracle1/completed.json',ROOT/'oracle_diag/results/v3_20261008/stage_a2/hopper_source_acceptance.json',
        OUT/'config.json',OUT/'authorization.json',OUT/'cache/raw_indices.npy',OUT/'acceptance/sample_buffer_indices.npy',OUT/'acceptance/sample_raw_indices.npy']
    write(OUT/'dependency_hashes.json',dict(sha256={str(p):sha(p) for p in refs}))
    write(OUT/'code_frozen.json',dict(sha256={str(p):sha(p) for p in HERE.glob('stage1_*.py')}))
    write(OUT/'PRE_RESULT_COMMITMENT.json',dict(config_sha256=sha(OUT/'config.json'),code_frozen_sha256=sha(OUT/'code_frozen.json'),
        sample_sha256=sha(OUT/'acceptance/sample_raw_indices.npy'),before_any_new_simulator_steps=True,time=time.time()))
    protected_check('before_simulator')
    print(json.dumps(dict(prepared=True,CPU_only=True,canonical=999998,independent_sample=10000,protected=len(protected))))

if __name__=='__main__':main()
