"""V12 0.1 only. CPU simulator replay; no learner, policy, or training entry point."""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MUJOCO_PY_FORCE_CPU'] = '1'
os.environ.pop('ODRL_PAPER_STYLE_SOURCE_TARGET', None)
sys.dont_write_bytecode = True

import argparse
import ast
import datetime
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback


HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stats(values):
    import numpy as np
    x = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(x)
    result = dict(n=int(x.size), finite_count=int(valid.sum()), missing_count=int((~valid).sum()))
    if not valid.any():
        return dict(result, maximum=None, median=None, p99=None, mean=None)
    a = x[valid]
    return dict(result, maximum=float(a.max()), median=float(np.median(a)),
                p99=float(np.percentile(a, 99, interpolation='linear')), mean=float(a.mean()))


def hash_tree(root):
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            assert root in path.resolve().parents, 'External symlink in protected package'
            result[path.relative_to(root).as_posix()] = sha(path)
    return result


def verify_project(root, expected_manifest):
    assert sha(root/'MANIFEST.json') == expected_manifest, 'Unexpected transfer manifest'
    entries = read(root/'MANIFEST.json')['files']
    hashes = {'MANIFEST.json': expected_manifest}
    for item in entries:
        path = (root/item['path']).resolve()
        assert root in path.parents and path.is_file(), 'Missing or external project input: ' + item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Input changed: ' + item['path']
        hashes[item['path']] = item['sha256']
    return hashes


def extract(root, relative, names, namespace, expected):
    tree = ast.parse((root/relative).read_text(encoding='utf-8-sig'), filename=str(root/relative))
    selected = []
    for name in names:
        found = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
        assert len(found) == 1, (relative, name)
        node = found[0]
        fingerprint = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        assert fingerprint == expected[relative+'::'+name], 'Restoration definition changed'
        selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(root/relative), 'exec'), namespace)


def load_restoration(root, expected):
    import numpy as np
    ns = dict(np=np, QVEL_CLIP=10.)
    extract(root, 'source_snapshot/oracle_diag/common.py',
            ['clips_qvel', 'obs_to_qpos_qvel', 'is_qvel_clipped'], ns, expected)
    extract(root, 'source_snapshot/oracle_diag/verify_replay_v2.py', ['step', 'recover'], ns, expected)
    return ns


def sample_indices(n, count, seed):
    import numpy as np
    return np.random.RandomState(seed).choice(n, size=count, replace=True).astype('<i8')


def resource_snapshot():
    result = dict(logical_cpu_count=os.cpu_count(), affinity_cpu_count=None, cgroups={})
    if hasattr(os, 'sched_getaffinity'):
        result['affinity_cpu_count'] = len(os.sched_getaffinity(0))
    for path in ('/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpuset.cpus.effective',
                 '/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current',
                 '/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.cfs_period_us',
                 '/proc/meminfo'):
        p = Path(path)
        if p.is_file():
            result['cgroups'][path] = p.read_text()
    if Path('/sys/fs/cgroup/cpu.max').is_file():
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        result['cpu_quota_cores'] = None if quota == 'max' else float(quota)/float(period)
    for label, cmd in (('nproc', ['nproc']), ('nvidia_smi_snapshot', ['nvidia-smi'])):
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
            result[label] = dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        except (OSError, subprocess.TimeoutExpired) as e:
            result[label] = dict(error=str(e))
    return result


def prepare_private_binding(out):
    # mujoco-py may write a build-lock or rebuild an extension during import.
    # Route all of those writes to a new copy; never import the installed package.
    original = Path(metadata.distribution('mujoco-py').locate_file('mujoco_py')).resolve()
    assert Path(sys.prefix).resolve() in original.parents and (original/'__init__.py').is_file()
    before = hash_tree(original)
    vendor = out/'private_runtime'
    vendor.mkdir()
    target = vendor/'mujoco_py'
    shutil.copytree(original, target)
    sys.path.insert(0, str(vendor))
    write(out/'MUJOCO_BINDING_COPY.json', dict(original=str(original), copy=str(target),
          original_hashes=before, original_files=len(before),
          copied_bytes=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()),
          reason='Confine import build-lock, patchelf, and any extension-build files to new output'))
    return original, before, target


class CountedEnvironment:
    """Count only calls routed through this adapter; delegate original physics unchanged."""
    def __init__(self, env):
        self.env = env
        self.steps = 0
        self.returned_steps = 0
        self.phase = 'initial'
        self.by_phase = {}
        self.original_step = env.step
        def counted(action):
            self.steps += 1
            self.by_phase[self.phase] = self.by_phase.get(self.phase, 0) + 1
            result = self.original_step(action)
            self.returned_steps += 1
            return result
        env.step = counted


def run_task(root, out, task, spec, conf, restoration, counts):
    import numpy as np
    import h5py
    from gym.envs.mujoco.hopper_v3 import HopperEnv
    from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
    tdir = out/task
    tdir.mkdir()
    dataset = root/spec['target_dataset']
    with h5py.File(dataset, 'r') as f:
        raw = {k: np.asarray(f[k]) for k in ('observations', 'actions', 'next_observations', 'rewards', 'terminals')}
    obs, actions, nexts = (raw[k] for k in ('observations', 'actions', 'next_observations'))
    rewards = raw['rewards'].reshape(-1)
    terminals = raw['terminals'].reshape(-1).astype(bool)
    n = len(obs)
    assert n == spec['target_rows'] == 5000
    assert obs.shape == nexts.shape == (n, spec['state_dim'])
    assert actions.shape == (n, spec['action_dim'])
    assert all(np.isfinite(a).all() for a in raw.values())
    # Same chronological boundary rule as the accepted Windows v2 stage_c.
    boundary = np.r_[np.max(np.abs(nexts[:-1]-obs[1:]), axis=1) > conf['boundary_abs_tolerance'], False]
    starts = np.r_[True, (boundary | terminals)[:-1]]
    assert np.flatnonzero(starts).tolist() == conf['expected_episode_starts'][task]
    sample = sample_indices(n, conf['sample_rows_per_task'], conf['sample_seeds'][task])
    np.save(str(tdir/'sample_indices.npy'), sample)
    write(tdir/'PRE_SIMULATOR_COMMITMENT.json', dict(time=utc(), sample_sha256=sha(tdir/'sample_indices.npy'),
        config_sha256=sha(out/'config.json'), target_dataset_sha256=sha(dataset), xml_sha256=sha(root/spec['xml']),
        sample_count=len(sample), unique_rows=int(np.unique(sample).size), sampling_with_replacement=True,
        training_updates=0, model_instances=0, optimizer_instances=0))
    constructor = HopperEnv if task == 'hopper-morph' else Walker2dEnv
    counts['environment_construction_attempts'] += 1
    env = constructor(xml_file=str(root/spec['xml']))
    counts['environment_instances'] += 1
    # Gym 0.18.3's byte-verified MujocoEnv.__init__ performs one self.step.
    counts['known_constructor_environment_steps'] += 1
    tracker = CountedEnvironment(env)
    start = time.monotonic()
    try:
        env.seed(conf['sample_seeds'][task])
        assert env.frame_skip == 4 and abs(float(env.model.opt.timestep)-.002) < 1e-12
        assert env.model.nq == (6 if task == 'hopper-morph' else 9)
        tracker.phase = 'chronological_state_recovery'
        p, v, status, recovery_error, clipped = restoration['recover'](env, obs, actions, starts, task)
        np.savez_compressed(str(tdir/'restoration_states.npz'), qpos=p, qvel=v, status=status,
                            recovery_error=recovery_error, clipped=clipped, episode_starts=starts)
        predicted = np.full((len(sample), spec['state_dim']), np.nan, np.float64)
        pred_r = np.full(len(sample), np.nan)
        pred_d = np.full(len(sample), -1, np.int8)
        state_error = np.full(len(sample), np.nan)
        reward_error = np.full(len(sample), np.nan)
        restored_error = np.full(len(sample), np.nan)
        failed_restore = []
        tracker.phase = 'sampled_one_step_replay'
        with (tdir/'replay_steps.jsonl').open('x', encoding='utf-8') as log:
            for j, row in enumerate(sample):
                row = int(row)
                if status[row] == 2:
                    failed_restore.append(dict(sample_position=j, data_row=row, reason='Verified chronological state restoration unavailable'))
                    log.write(json.dumps(dict(sample_position=j, data_row=row, restored=False))+'\n')
                    continue
                ns, reward, done, _, _ = restoration['step'](env, p[row], v[row], actions[row])
                assert np.isfinite(ns).all() and np.isfinite(reward), 'Non-finite replay result'
                predicted[j] = ns
                pred_r[j], pred_d[j] = reward, done
                state_error[j] = np.max(np.abs(ns-nexts[row]))
                reward_error[j] = abs(reward-rewards[row])
                # Current observation reconstruction error is recorded separately.
                restored_error[j] = float(recovery_error[row])
                log.write(json.dumps(dict(sample_position=j, data_row=row, restored=True,
                    state_linf=float(state_error[j]), reward_abs=float(reward_error[j]),
                    terminal_matches=bool(done == terminals[row])), allow_nan=False)+'\n')
                if (j+1) % 1000 == 0:
                    log.flush()
                    print(task, 'replayed', j+1, '/', len(sample), flush=True)
        np.savez_compressed(str(tdir/'replay_inputs_outputs.npz'), sample_indices=sample,
            reference_next_state=nexts[sample], replay_next_state=predicted,
            reference_reward=rewards[sample], replay_reward=pred_r,
            reference_terminal=terminals[sample], replay_terminal=pred_d,
            state_error=state_error, reward_error=reward_error,
            recovery_error=restored_error, recoverable=status[sample] != 2)
        finite = np.isfinite(state_error)
        done_mismatches = int(np.count_nonzero((pred_d != terminals[sample]) & finite))
        old = read(HERE/'references'/conf['windows_reference_files'][task])
        ss, rs = stats(state_error), stats(reward_error)
        # Historical hard limits are reported only as a diagnostic, NOT a new V12 acceptance rule.
        historical_limits_met = bool(finite.all() and (state_error <= 1e-3).all()
            and (reward_error <= 1e-4).all() and done_mismatches == 0)
        ratio = dict(maximum=None if ss['maximum'] is None else ss['maximum']/old['state_error']['max'],
                     median=None if ss['median'] is None else ss['median']/old['state_error']['median'])
        result = dict(task=task, sample_count=len(sample), unique_sampled_rows=int(np.unique(sample).size),
            dataset_rows=n, dataset_clipped_rows=int(clipped.sum()),
            dataset_state_restoration_failures=int((status == 2).sum()),
            dataset_failed_restoration_rows=np.flatnonzero(status == 2).tolist(),
            sampled_state_restoration_failures=len(failed_restore), failed_restorations=failed_restore,
            state_linf_error=ss, reward_abs_error=rs, terminal_mismatches=done_mismatches,
            historical_state_limit=1e-3, historical_reward_limit=1e-4,
            historical_limits_met=historical_limits_met,
            windows_reference_state_error=old['state_error'], state_error_ratio_to_windows=ratio,
            windows_reference_same_dataset=sha(dataset) == old['source_sha256'],
            windows_reference_same_xml=sha(root/spec['xml']) == old['xml_sha256'],
            cross_platform_bitwise_equality_required=False, v12_replay_accepted=None,
            acceptance_status='AWAITING_USER_REPLAY_REVIEW',
            reason='V12 does not quantify significant replay-error increase; no automatic passage to 0.2',
            known_constructor_environment_steps=1, adapter_environment_step_attempts=tracker.steps,
            adapter_environment_step_returns=tracker.returned_steps, steps_by_phase=tracker.by_phase,
            env_class=type(env).__module__+'.'+type(env).__name__, timestep=float(env.model.opt.timestep),
            frame_skip=env.frame_skip, elapsed_seconds=time.monotonic()-start,
            no_policy_evaluation=True, no_oracle_cache_created=True, new_training_updates=0)
        assert result['windows_reference_same_dataset'] and result['windows_reference_same_xml']
        write(tdir/'RESULT.json', result)
        return result
    finally:
        counts['adapter_environment_step_attempts'] += tracker.steps
        counts['adapter_environment_step_returns'] += tracker.returned_steps
        env.close()


def render_report(out, results, counts, failure=None):
    rows = ['# V12 阶段 0.1 环境运行验收报告', '',
            '本次仅 CPU 状态恢复与一步回放，不训练、不创建学习器、不进行策略评估，不进入 0.2。', '',
            '每任务从 5000 行中按预先固定种子有放回抽 10000 次，重复行不是新增独立样本。',
            '截断速度沿用旧验收的已验证轨迹历史恢复方法；不把截断观测直接当成完整物理状态。', '',
            '|任务|回放次数|独立数据行|状态误差最大值|中位数|99分位|抽样恢复失败数|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for task, x in results.items():
        e = x['state_linf_error']
        fmt = lambda v: '未得到' if v is None else '%.9g' % v
        rows.append('|%s|%d|%d|%s|%s|%s|%d|' % (task, x['sample_count'], x['unique_sampled_rows'],
            fmt(e['maximum']), fmt(e['median']), fmt(e['p99']), x['sampled_state_restoration_failures']))
    for task, x in results.items():
        rows += ['', '## '+task, '',
                 'Windows 原全 5000 行状态误差统计：'+json.dumps(x['windows_reference_state_error'], ensure_ascii=False),
                 'Linux 抽样最大值和中位数相对 Windows 比值：'+json.dumps(x['state_error_ratio_to_windows'], ensure_ascii=False),
                 '恢复失败的全数据行数：%d；旧技术容差内：%s。旧技术容差不是 V12 新的自动放行阈值。' %
                 (x['dataset_state_restoration_failures'], x['historical_limits_met']),
                 '实际环境步分类：'+json.dumps(x['steps_by_phase'], ensure_ascii=False)+'；另有构造器 1 步。',
                 '回放误差逐样本与原始对照保存在该任务子目录；失败样本保留，不通过剔除样本改变统计。']
    rows += ['', '## 停止位置', '',
             'AWAITING_USER_REPLAY_REVIEW。请先审阅 Linux 与 Windows 的误差及恢复失败记录，确认后才能准备或启动 0.2。',
             'V12 的“明显变大”没有数值门槛，本入口不自行补写放行判定。',
             '新增训练更新 0；GPU 测试 0；已发生的环境步计数：'+json.dumps(counts, ensure_ascii=False),
             '本次输入、环境盘点及保护核对见对应 JSON；旧迁移项目与原 Conda 的 mujoco_py 目录在运行前后核对。',
             '当前不运行阶段 1A、1B、ROOT 阶段 2 或阶段 3，也不运行 SA-top 正式训练。']
    if failure:
        rows += ['', '## 故障', '', '已停止，不自动重试。', '```', failure, '```']
    with (out/'REPORT_ZH.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(rows)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert sys.version_info[:3] == (3, 8, 13), 'Use the accepted Python 3.8.13 environment'
    assert sys.platform == 'linux', 'This is a Linux-only simulator acceptance command'
    assert os.environ.get('TMUX'), 'Run inside tmux as required by V12'
    assert Path(sys.prefix).resolve() == (Path.home()/'conda-envs/offdynamics38').resolve()
    root, out = args.project.resolve(), args.output.resolve()
    expected_parent = (Path.home()/'v12_linux/outputs').resolve()
    assert out.parent == expected_parent and out.name.startswith('stage0_1_'), 'Use the prescribed new output directory'
    assert root not in out.parents and root != out
    freeze = read(HERE/'FROZEN.json')
    for name, digest in freeze['files'].items():
        assert sha(HERE/name) == digest, 'V12 package changed: '+name
    out.mkdir(parents=True, exist_ok=False)
    counts = dict(environment_construction_attempts=0, environment_instances=0,
                  known_constructor_environment_steps=0, adapter_environment_step_attempts=0,
                  adapter_environment_step_returns=0)
    results, protected, original_binding, binding_before = {}, None, None, None
    conda_meta_before = None
    begin = time.monotonic()
    failure = None
    try:
        conf = read(HERE/'config_template.json')
        write(out/'config.json', dict(conf, actual_project=str(root), actual_output=str(out),
                                     TMUX=os.environ['TMUX'], started_utc=utc()))
        print('Checking frozen transfer project before simulator construction', flush=True)
        protected = verify_project(root, freeze['project_manifest_sha256'])
        write(out/'INPUT_HASHES_BEFORE.json', protected)
        write(out/'RESOURCES_BEFORE.json', resource_snapshot())
        mu = Path(os.environ['MUJOCO_PY_MUJOCO_PATH']).expanduser().resolve()
        assert (mu/'bin/libmujoco210.so').is_file(), 'MuJoCo 2.1 shared library missing'
        for name, version in conf['expected_versions'].items():
            assert metadata.version(name) == version, 'Unexpected package version: '+name
        conda_meta = Path(sys.prefix)/'conda-meta'
        conda_meta_before = hash_tree(conda_meta)
        write(out/'CONDA_METADATA_HASHES_BEFORE.json', conda_meta_before)
        write(out/'CONDA_PACKAGES.json', [read(p) for p in sorted(conda_meta.glob('*.json'))])
        for name in ('temporary_files', 'cache'):
            (out/name).mkdir()
        os.environ['TMPDIR'] = str(out/'temporary_files')
        os.environ['XDG_CACHE_HOME'] = str(out/'cache')
        os.environ['CYTHON_CACHE_DIR'] = str(out/'cache/cython')
        inventory = dict(python=sys.version, executable=sys.executable, prefix=sys.prefix,
            platform=platform.platform(), libc=platform.libc_ver(),
            packages={dist.metadata['Name']: dist.version for dist in metadata.distributions()},
            variables={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'MUJOCO_PY_MUJOCO_PATH',
                'MUJOCO_PY_FORCE_CPU','OMP_NUM_THREADS','MKL_NUM_THREADS','LD_LIBRARY_PATH')},
            mujoco_library_sha256=sha(mu/'bin/libmujoco210.so'))
        write(out/'ENVIRONMENT.json', inventory)
        gym_root = Path(metadata.distribution('gym').locate_file('gym')).resolve()
        gym_before = {}
        for rel in ('envs/mujoco/hopper_v3.py', 'envs/mujoco/walker2d_v3.py',
                    'envs/mujoco/mujoco_env.py','wrappers/time_limit.py','utils/seeding.py','utils/ezpickle.py'):
            actual = gym_root/rel
            assert sha(actual) == sha(root/'environment/gym_source_reference'/rel), 'Gym implementation changed: '+rel
            gym_before[str(actual)] = sha(actual)
        write(out/'GYM_INPUT_HASHES.json', gym_before)
        original_binding, binding_before, private_binding = prepare_private_binding(out)
        import mujoco_py
        assert private_binding in Path(mujoco_py.__file__).resolve().parents
        import numpy as np
        expected_ast = read(HERE/'RESTORATION_AST.json')
        restoration = load_restoration(HERE/'references', expected_ast)
        mapping = read(root/'cluster/paths.example.json')['tasks']
        for task in conf['task_order']:
            print('Starting CPU replay acceptance:', task, flush=True)
            results[task] = run_task(root, out, task, mapping[task], conf, restoration, counts)
            # Do not proceed to another task after a concrete restoration/numerical failure.
            if not results[task]['historical_limits_met']:
                break
        assert 'torch' not in sys.modules, 'Unexpected learner stack import'
        after = verify_project(root, freeze['project_manifest_sha256'])
        assert after == protected
        assert hash_tree(original_binding) == binding_before, 'Original Conda mujoco_py files changed'
        assert hash_tree(conda_meta) == conda_meta_before, 'Conda package metadata changed'
        assert sha(mu/'bin/libmujoco210.so') == inventory['mujoco_library_sha256'], 'MuJoCo shared library changed'
        for path, h in gym_before.items():
            assert sha(path) == h
        write(out/'INPUT_PROTECTION_AFTER.json', dict(passed=True, transfer_files=len(after),
              original_mujoco_binding_files=len(binding_before), gym_reference_files=6,
              conda_metadata_files=len(conda_meta_before), mujoco_library_unchanged=True))
        render_report(out, results, counts)
        passed_technical_checks = len(results) == 2 and all(x['historical_limits_met'] for x in results.values())
        receipt = dict(completed=True, acceptance_status='AWAITING_USER_REPLAY_REVIEW',
            v12_environment_acceptance_approved=False, historical_technical_checks=passed_technical_checks,
            task_results={t:str(out/t/'RESULT.json') for t in results}, counts=counts,
            new_training_updates=0, model_instances=0, optimizer_instances=0, GPU_tests=0,
            Torch_imported=False, stage0_2_started=False, later_stages_started=False,
            elapsed_seconds=time.monotonic()-begin, ended_utc=utc())
        receipt['evidence_sha256'] = {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*')
            if p.is_file() and 'private_runtime' not in p.relative_to(out).parts}
        write(out/'STAGE0_1_COMPLETED.json', receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        print('Stopped. Read:', out/'REPORT_ZH.md', flush=True)
        return 0 if passed_technical_checks else 2
    except BaseException:
        failure = traceback.format_exc()
        protection = {}
        if protected is not None:
            try:
                protection['transfer_unchanged'] = verify_project(root, freeze['project_manifest_sha256']) == protected
            except BaseException:
                protection['transfer_unchanged'] = False
        if original_binding is not None and binding_before is not None:
            protection['original_binding_unchanged'] = hash_tree(original_binding) == binding_before
        if conda_meta_before is not None:
            protection['conda_metadata_unchanged'] = hash_tree(Path(sys.prefix)/'conda-meta') == conda_meta_before
        write(out/'FAILED.json', dict(time=utc(), traceback=failure, counts=counts,
            input_protection=protection, new_training_updates=0, automatic_retry=False,
            stage0_2_started=False, later_stages_started=False))
        if not (out/'REPORT_ZH.md').exists():
            render_report(out, results, counts, failure)
        print(failure, file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
