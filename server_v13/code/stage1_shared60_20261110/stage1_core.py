"""V13 Stage 1 paths, input receipts and read-only artifact checks."""
import json
from pathlib import Path
import sys

from common import read, sha


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PROJECT = (Path.home()/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104').resolve()
STAGE0 = BASE/'stage0'
STAGE0_REPAIRED = BASE/'outputs/stage0_20261110_report_repaired'
CONFIG = read(HERE/'config.json')
TASK = CONFIG['task']
SUCCESS = {'observation_noise_floor', 'projected_gradient', 'step_tolerance', 'cost_tolerance'}
GIB = 1024 ** 3


def publish(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def verify_stage0():
    expected = CONFIG['original_stage0_repaired_evidence']
    receipt = read(STAGE0_REPAIRED/'STAGE0_COMPLETED.json')
    assert receipt['completed'] and receipt['target_control_passed'] and receipt['report_recovery_only']
    assert receipt['source_count'] == 20000 and receipt['target_count'] == 5000 and not receipt['later_stage_started']
    assert receipt['acceptance_status'] == 'AWAITING_STAGE1_CONFIRMATION' and receipt['new_simulator_steps'] == 0
    assert receipt['evidence_sha256'] == expected
    for name, digest in expected.items():
        assert sha(STAGE0_REPAIRED/name) == digest, ('Changed stage 0 receipt artifact', name)
    assert sha(STAGE0/'FROZEN.json') == CONFIG['original_stage0_freeze_sha256']
    for name,digest in read(STAGE0/'FROZEN.json')['files'].items():
        assert sha(STAGE0/name) == digest, ('Changed original stage 0 package', name)
    return {name: digest for name,digest in expected.items()}


def old_solver_config():
    verify_stage0()
    cfg = read(STAGE0/'config.json')
    assert cfg['action_boundary_abs_tolerance'] == CONFIG['cache']['boundary_abs_tolerance']
    return cfg


def cache_receipt(cache, *, complete=False):
    record = read(cache/'CACHE_ACCEPTANCE.json')
    assert record['passed'] and record['rows'] == CONFIG['canonical_source_rows']
    assert record['replay_count'] == CONFIG['cache']['audit_rows']
    assert record['filter'] == CONFIG['residual_filter'] and record['stage2_authorized'] is False
    if complete:
        verify_cache_files(cache, record, record['cache_sha256'])
        assert sha(cache/'CACHE_REPLAY_RAW.npz') == record['replay_sha256']
        assert sha(cache/'CACHE_STATISTICS.json') == record['statistics_sha256']
    return record


def verify_cache_files(cache, receipt, names):
    for name in names:
        assert sha(cache/name) == receipt['cache_sha256'][name], ('Changed accepted cache', name)


def verify_worker(job_path, stage):
    cfg = read(job_path)
    run = Path(cfg['run']).resolve()
    assert run.parent == BASE/'outputs' and cfg['stage'] == stage == 1
    plan = read(run/'PLAN.json')
    assert sha(run/'PLAN.json') == cfg['plan_sha256']
    assert cfg['job'] in plan['job_ids']
    assert not plan['stage2_authorized']
    assert sha(run/'AUTHORIZATION.json') == plan['authorization_sha256']
    assert sha(HERE/'FROZEN.json') == plan['code_freeze_sha256']
    return cfg


def verify_runtime():
    import importlib.metadata as metadata
    assert sys.platform == 'linux' and sys.version_info[:3] == (3,8,13)
    assert Path(sys.prefix).resolve() == (Path.home()/'conda-envs/offdynamics38').resolve()
    expected = dict(numpy='1.23.5', h5py='3.11.0', scipy='1.4.1', gym='0.18.3', **{'mujoco-py':'2.1.2.14'})
    for name, version in expected.items():
        assert metadata.version(name) == version, ('Unexpected package version', name)
    project = read(PROJECT/'MANIFEST.json')
    manifest = {entry['path']:entry['sha256'] for entry in project['files']}
    gym = Path(metadata.distribution('gym').locate_file('gym')).resolve()
    files = {}
    for name in ('envs/mujoco/hopper_v3.py','envs/mujoco/mujoco_env.py','wrappers/time_limit.py','utils/seeding.py','utils/ezpickle.py'):
        p = gym/name
        files[str(p)] = sha(p)
        assert files[str(p)] == manifest['environment/gym_source_reference/'+name], ('Changed installed Gym file', name)
    return files
