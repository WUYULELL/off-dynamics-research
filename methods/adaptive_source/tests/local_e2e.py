"""Local end-to-end test of the adaptive-source controllers on synthetic data (CPU, fake environment).

Builds small-config copies of the jobs in <work>/ (shared files byte-identical to src/), writes the
ASRC_LOCAL_TEST json and runs controller.py of j013 -> j015a -> j014 (one job) -> j015b -> j016.
usage: python local_e2e.py <work_dir> <venv python> <ot python> [stage ...]
"""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import build_jobs as B  # noqa: E402

REPO = HERE.parent.parent.parent
T1, T2 = 'hopper-kinematic-legjnt-medium', 'walker2d-gravity-2.0'
COMMON = dict(max_per_gpu=4, mem_limit=0.95, external_retry_max=1, launch_gap_seconds=0)


def main():
    work = Path(sys.argv[1]).resolve()
    py, ot_py = sys.argv[2], sys.argv[3]
    stages = sys.argv[4:] or ['data', 'j013', 'j015a', 'j014', 'j015b', 'j016']
    work.mkdir(parents=True, exist_ok=True)
    data = work/'data'
    if 'data' in stages:
        subprocess.run([py, str(HERE/'make_synthetic.py'), str(data), '20000', '2000'], check=True)
    files = json.loads((data/'files.json').read_text())
    vendor = work/'vendor'
    vendor.mkdir(exist_ok=True)
    hm = files['target_files']['hopper-morph']['path']
    dirs = {n: str(work/n/'code') for n in ('j013_adapt_preflight', 'j015a_dyn_models', 'j014_s3129_hoplegm', 'j015b_offline_select', 'j016_sim_check')}
    local = dict(files, vendor=str(vendor), algo=str(REPO/'algo'), ot_python=ot_py, env_module_name='fake_env',
                 otdf_job_overrides=dict(env_factory='fake_env', device='cpu'), bc_job_overrides=dict(env_factory='fake_env', device='cpu'),
                 test_job_overrides=dict(device='cpu'), dyn_source_overrides=dict(device='cpu'), dyn_target_overrides=dict(device='cpu'),
                 preflight_dir=dirs['j013_adapt_preflight'],
                 dirs=dict(dirs, dyn=dirs['j015a_dyn_models'], select=dirs['j015b_offline_select']),
                 select_overrides={T1: dict(device='cpu')}, sim_overrides={T1: dict(device='cpu', env_factory='fake_env')})
    lcfg = work/'local.json'
    lcfg.write_text(json.dumps(local, indent=2))
    layout = {'hoplegm': {'3129': {c: 'j014_s3129_hoplegm' for c in B.CANDIDATES+['rho80T', 'rho80R']}}}
    hm_extra = {'otdf_official/datasets/hopper_morph_medium.hdf5': hm}
    cfgs = {
        'j013_adapt_preflight': dict(job='j013_adapt_preflight', kind='preflight', tasks=[T1, T2], **COMMON, **B.LAYER,
                                     ot=dict(conda_env_candidates=[], cores_per_process=1, max_hours=1.0),
                                     unit_tests=dict(steps=20, a_tasks=[T1], layer_tasks=[T1]),
                                     smoke=dict(seed=2999, steps=100, iqlcal_tasks=[T1]),
                                     throughput=dict(tasks=[T2], steps=200, eval_every=100),
                                     normalization_grid=[-100, 0, 1000, 3000], support_definition=B.SUPPORT_DEFINITION),
        'j015a_dyn_models': dict(job='j015a_dyn_models', kind='dyn', tasks=[T1], seeds=[3129, 3130], **COMMON,
                                 model=dict(members=5, hidden=[200, 200, 200, 200],
                                            source=dict(steps=300, batch=512, lr=1e-3, validation_fraction=0.02, eval_every=100),
                                            target=dict(max_steps=200, batch=256, lr=3e-4, eval_every=50, patience_evals=2),
                                            folds_max=5, es_fraction=0.1, eps0=[0.05, 0.1, 0.2, 0.5], quantile=0.9)),
        'j014_s3129_hoplegm': dict(job='j014_s3129_hoplegm', kind='train', tasks=[T1], seeds=[3129], seed_origin=3129,
                                   groups=[B.GROUPS[g] for g in B.ORDER7], otdf=dict(steps=200, eval_every=40, save_last=5),
                                   bc=dict(steps=200, eval_every=40, save_last=5), preflight_candidates=B.PREFLIGHT, **COMMON, **B.LAYER),
        'j015b_offline_select': dict(job='j015b_offline_select', kind='select', tasks=[T1], seeds=[3129], candidates=B.CANDIDATES,
                                     default='rho80', kappas=[0, 0.5, 1, 1.5, 2, 3], fixed_horizons=[50, 100, 200, 1000],
                                     starts=dict(trajectory=50, uniform=50), j014_layout=layout, dyn_candidates=B.DYN,
                                     preflight_candidates=B.PREFLIGHT, **COMMON),
        'j016_sim_check': dict(job='j016_sim_check', kind='simcheck', tasks=[T1], seeds=[3129], candidates=B.CANDIDATES,
                               fixed_horizons=[50, 100, 200, 1000], j014_layout=layout, dyn_candidates=B.DYN, select_candidates=B.SELECT,
                               preflight_candidates=B.PREFLIGHT, wait_hours=0.01, **COMMON),
    }
    env = dict(os.environ, ASRC_LOCAL_TEST=str(lcfg), PYTHONPATH=str(HERE), PYTHONDONTWRITEBYTECODE='1',
               D4RL_SUPPRESS_IMPORT_ERROR='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='20261127',
               PATH=str(Path(py).parent)+os.pathsep+os.environ.get('PATH', ''))
    for name in ('j013_adapt_preflight', 'j015a_dyn_models', 'j014_s3129_hoplegm', 'j015b_offline_select', 'j016_sim_check'):
        short = name.split('_')[0] if not name.startswith('j014') else 'j014'
        if short not in stages:
            continue
        job = B.job_json(name, 'local test', 'local test', 1, None)
        B.build(work/name, job, cfgs[name], hm_extra if name in ('j013_adapt_preflight', 'j015a_dyn_models') else None)
        print('==== running', name, flush=True)
        p = subprocess.run([py, '-u', '-B', str(work/name/'code'/'controller.py')], env=env, cwd=str(work/name/'code'))
        print('==== %s exit code %d' % (name, p.returncode), flush=True)
        if p.returncode != 0 and name != 'j013_adapt_preflight':
            sys.exit(p.returncode)


if __name__ == '__main__':
    main()
