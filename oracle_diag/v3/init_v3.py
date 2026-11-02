"""Freeze the reviewed v3 protocol before any diagnostic or training results."""
from support_v3 import *
import time
from task_env import normalization,paths,TASKS

def main():
    fresh(OUT)
    protected=[]
    for base in ['dataset','envs','config','algo','oracle_diag']:
        for path in (ROOT/base).rglob('*'):
            if HERE in path.parents or OUT in path.parents:continue
            if os.path.isfile(winpath(path)):protected.append(path)
    protected.append(ROOT/'train.py')
    # Explicit source datasets and task assets are covered even when outside these trees.
    for kind in ['walker2d','hopper']:protected.append(Path.home()/'.d4rl/datasets'/(kind+'_medium-v2.hdf5'))
    for task in TASKS:protected.extend(paths(task))
    protected.append(ROOT/'reproductions/OTDF/info.py')
    hashes={str(path):sha(path) for path in sorted(set(protected))}
    write(OUT/'protected_manifest.json',dict(created=time.time(),sha256=hashes,count=len(hashes)))
    source=Path('D:/claude 离线/给Code的指令_v3_标准方向.md')
    (OUT/'USER_SPEC_v3.md').write_bytes(source.read_bytes())
    write(OUT/'protocol.json',dict(created=time.time(),request_sha256=sha(source),
        scope='A: adapters, reused shifted target replay, new normal source replay and source oracle cache. B: Walker-morph 5 seeds. C not executed.',
        authorization='User requested task completion. User document explicitly requires confirmation before GPU/RL training; stage A and implementation proceed, training approval is requested after readiness.',
        review=['v2 did not establish a theoretical target-only upper bound or that source data are intrinsically useless',
                'v2 total draws were already equal: 256 target versus 128 source + 128 target',
                'MIX-NAIVE and MIX-CONST are identical as specified; one control and an identity check, no duplicate runs',
                'Existing v2 shifted replay evidence is reused without rerunning its 5000 rows',
                'Target replay is not the source oracle cache; source qpos/qvel are mapped by D4RL retained raw indices',
                'Oracle returns are not a mathematical performance ceiling; failed gates are limited evidence'],
        stage_a=dict(tasks=list(TASKS),primary='walker2d-morph',source_replay_environments=['walker2d','hopper'],
            sample_n=5000,sample_seed=0,state_tolerance=1e-3,reward_tolerance=1e-4,require_all_sampled_rows=True,
            source_raw_states='Full float64 infos/qpos and infos/qvel; privileged simulator state',
            source_oracle_tasks=['walker2d-morph'],source_oracle_scope='All retained normal D4RL source transitions; cold reset and zero warmstart',
            independent_validation_limit='Separate factual-domain replays do not independently validate counterfactual source-to-shifted transitions'),
        stage_b=dict(task='walker2d-morph',methods=['TARGET','MIX-CONST','MIX-ORACLE-BACKUP'],
            aliases={'MIX-NAIVE':'MIX-CONST'},seeds=list(range(5)),steps=500000,total_batch=128,
            target_only_draws=128,mix_source_draws=64,mix_target_draws=64,source_weight=.20,
            gamma=.99,eta_unchanged=1.,target_fraction=1.,paper_style_swap=False,
            loss='target_loss.mean() + 0.20 * source_loss.mean(); same V/actor weighting for both mixed arms; only source Q labels change',
            initialization='Shared seed and first-update RNG reset; aligned paired samplers',
            main_comparison='MIX-ORACLE-BACKUP minus TARGET',
            added_value_comparison='MIX-ORACLE-BACKUP minus MIX-CONST (MIX-NAIVE alias)',
            positive_gate='paired mean > 2*sample_SD/sqrt(5) and at least 4 positive seeds',
            limited_gate='mean > 0 and positive gate not passed',
            nonpositive_gate='mean <= 0: no positive observed mean; not proof of a universal impossibility',
            evaluation_steps=[460000,470000,480000,490000,500000],eval_episodes=10,
            evaluation_domain='shifted OTDF XML, TimeLimit 1000',normalization=normalization('walker2d-morph'),
            C='not executed; future separate decision'),normalization={t:normalization(t) for t in TASKS}))
    write(OUT/'status.json',dict(state='stage_a_preparing',gpu_training_authorized=False))
    print('V3 protocol frozen; protected files',len(hashes),flush=True)

if __name__=='__main__':main()
