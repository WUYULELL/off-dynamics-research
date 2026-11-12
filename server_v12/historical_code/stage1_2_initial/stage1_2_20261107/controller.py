"""One Linux controller owns all authorized V12 stage 1A, 1B and 2 subprocesses."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from common import (check_environment_protection, existing_environment_snapshot,
                    full_smi, gpu_info, idle_gpu, read, sha, utc, verify_package, verify_project, write)
from report import bc_report, cpu_lines
from resources import Monitor, busy_percent, cpu_ticks, limits_snapshot, tier_allowed
from science import HERE, Project, select_trajectories


HOME_DIR = Path.home()
OUTPUTS = HOME_DIR / 'v12_linux/outputs'
PROJECT = HOME_DIR / 'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
STAGE0 = OUTPUTS / 'stage0_3_20261107'
ROOT_PREFIX = HOME_DIR / 'conda-envs/v12_root_20261107'
STAGE_OUTPUTS = {'1A': 'stage1A_20261107', '1B': 'stage1B_20261107', '2': 'stage2_root_20261107'}
CAP = 6


def stage0_verified():
    path = STAGE0 / 'STAGE0_3_COMPLETED.json'
    result = read(path)
    assert result['completed'] and result['all_planned_tiers_measured']
    assert result['workers_completed_and_CPU_audited'] == 14 and not result['later_stages_started']
    assert result['measured_tiers'] == ['IQL_n1', 'IQL_n2', 'IQL_n4', 'IQL_n6', 'BC_n1']
    for name, digest in result['evidence_sha256'].items():
        evidence = (STAGE0 / name).resolve()
        assert STAGE0.resolve() in evidence.parents and sha(evidence) == digest, name
    assert read(STAGE0 / 'INPUT_PROTECTION_AFTER.json')['passed']
    return dict(stage0_3_receipt_sha256=sha(path), verified_evidence_files=len(result['evidence_sha256']),
                selected_GPU_UUID=result['physical_GPU_UUID'], stage0_user_accepted=True)


def process_env(gpu=None, prefix=None, jax=False):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu or '-1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', MUJOCO_PY_FORCE_CPU='1')
    for key in ('ODRL_PAPER_STYLE_SOURCE_TARGET', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    if prefix is not None:
        previous = str(Path(sys.prefix).resolve())
        for key in ('PATH', 'LD_LIBRARY_PATH', 'CPATH', 'LIBRARY_PATH'):
            env[key] = env.get(key, '').replace(previous, str(prefix))
        env['PATH'] = str(prefix / 'bin') + ':' + env['PATH']
        env['CONDA_PREFIX'] = str(prefix)
    if jax:
        nvidia = prefix / 'lib/python3.8/site-packages/nvidia'
        cuda = nvidia / 'cuda_nvcc'
        assert (cuda / 'bin/ptxas').is_file() and (cuda / 'nvvm/libdevice/libdevice.10.bc').is_file()
        env['PATH'] = str(cuda / 'bin') + ':' + env['PATH']
        libs = [str(path) for path in sorted(nvidia.glob('*/lib'))]
        env['LD_LIBRARY_PATH'] = ':'.join(libs + [env['LD_LIBRARY_PATH']])
        env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_PLATFORMS='cuda',
                   XLA_FLAGS=f'--xla_gpu_cuda_data_dir={cuda} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    return env


def monitor_jobs(base, names, commands, environments, limits, config, gpu, label):
    assert len(names) == len(commands) == len(environments) and 1 <= len(names) <= CAP
    monitor = Monitor(limits, config['resource_limits'], lambda: gpu_info(gpu['uuid']))
    processes, streams = [], []
    start = time.monotonic()
    try:
        for name, cmd, env in zip(names, commands, environments):
            runtime = base / 'runtime' / name
            for child in ('tmp', 'cache', 'cuda'):
                (runtime / child).mkdir(parents=True, exist_ok=False)
            env = dict(env, TMPDIR=str(runtime / 'tmp'), XDG_CACHE_HOME=str(runtime / 'cache'),
                       CYTHON_CACHE_DIR=str(runtime / 'cache' / 'cython'), CUDA_CACHE_PATH=str(runtime / 'cuda'))
            stdout = (base / 'logs' / (name + '.stdout.log')).open('x', encoding='utf-8')
            stderr = (base / 'logs' / (name + '.stderr.log')).open('x', encoding='utf-8')
            streams.extend([stdout, stderr])
            process = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, cwd=base)
            processes.append(process)
            write(base / 'logs' / (name + '.launch.json'), dict(pid=process.pid, parent=os.getpid(),
                  command=cmd, CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'], time=utc()))
        last_print = 0.
        with (base / 'logs' / (label + '.resource.jsonl')).open('x', encoding='utf-8') as file:
            while True:
                statuses = [p.poll() for p in processes]
                for name, status in zip(names, statuses):
                    if status not in (None, 0):
                        raise RuntimeError(f'{name} exited with code {status}. See {base / "logs" / (name + ".stderr.log")}')
                sample = monitor.sample(processes, label)
                file.write(json.dumps(sample, allow_nan=False) + '\n')
                file.flush()
                if all(status == 0 for status in statuses):
                    break
                now = time.monotonic()
                if now - last_print >= 30:
                    print(base.name, label, 'active PIDs:', [p.pid for p in processes if p.poll() is None], flush=True)
                    last_print = now
                time.sleep(config['monitor_interval_seconds'])
        host = [item['host_CPU_busy_percent'] for item in monitor.observed if item['host_CPU_busy_percent'] is not None]
        owned = [item['owned_CPU_percent_one_core_100'] for item in monitor.observed]
        return dict(wall_seconds=time.monotonic() - start, peak_worker_RSS_bytes=monitor.max_worker_RSS,
                    peak_gpu_used_MiB=max(item['gpu']['memory_used_MiB'] for item in monitor.observed),
                    sampled_mean_host_CPU_percent=sum(host) / len(host) if host else None,
                    sampled_peak_host_CPU_percent=max(host) if host else None,
                    sampled_mean_worker_CPU_percent=sum(owned) / len(owned),
                    sample_count=len(monitor.observed), pressure_warnings=monitor.warnings)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in streams:
            stream.close()


def launch_python(script, mode, path, prefix=None):
    binary = sys.executable if prefix is None else str(prefix / 'bin/python')
    return [binary, '-u', '-B', str(HERE / script), mode, str(path)]


def inventory(base, start, limits, host_start, affinity_start, session, launch):
    result = dict(limits=limits, elapsed_seconds=time.monotonic() - start, tmux_session=session, launch=launch,
                  host_CPU_busy_percent_entire_stage=busy_percent(host_start, cpu_ticks()),
                  affinity_CPU_busy_percent_entire_stage=busy_percent(affinity_start, cpu_ticks(limits['affinity'])))
    write(base / 'RESOURCE_SUMMARY.json', result)
    return result


def make_bc_jobs(stage, base, gpu):
    jobs = []
    groups = [('hopper-morph', 'source')] if stage == '1A' else [
        (task, size) for task in ('hopper-morph', 'walker2d-morph') for size in (2, 4, 7)]
    for task, size in groups:
        for seed in range(401, 406):
            name = f'{task}_{size}_s{seed}'
            run = base / 'runs' / name
            run.mkdir(parents=True, exist_ok=False)
            jobs.append(dict(id=name, task=task, trajectories=size, seed=seed, stage=stage, steps=200000,
                             gpu_uuid=gpu['uuid'], project=str(PROJECT), base=str(base), output=str(run),
                             controller_pid=os.getpid()))
    return jobs


def run_bc(stage, base, config, gpu):
    project = Project(PROJECT)
    if stage == '1B':
        write(base / 'TRAJECTORY_SUBSETS.json', select_trajectories(project))
    (base / 'jobs').mkdir()
    jobs = make_bc_jobs(stage, base, gpu)
    paths = {job['id']: base / 'jobs' / (job['id'] + '.json') for job in jobs}
    for job in jobs:
        write(paths[job['id']], job)
    write(base / 'AUTHORIZATION.json', dict(stage=stage, jobs={name: sha(path) for name, path in paths.items()},
          total_jobs=len(jobs), max_GPU_training_processes=CAP, next_stage_authorized=True))
    write(base / 'config.json', dict(stage=stage, seed_range=[401, 405], steps_per_job=200000, cosine_T_max=200000,
          evaluation_every=5000, episode_count=10, seeds_fixed=True, maximum_GPU_workers=CAP, Torch_threads=1,
          BLAS_threads=1, lr=.0003, batch=64, beta=0, trajectory_seed=777 if stage == '1B' else None,
          rule='group Last5<20 and pooled Last5 length<200 fraction>0.5' if stage == '1A' else
          'delta(7−4)>2*WelchSE unsaturated; delta<0 or delta<WelchSE saturated; otherwise uncertain',
          last5_points=[180000, 185000, 190000, 195000, 200000], gpu_uuid=gpu['uuid']))
    waves = []
    rss, gpu_per_worker = None, None
    remaining = list(jobs)
    while remaining:
        limits = limits_snapshot()
        cap = 2 if stage == '1A' and not waves else CAP
        for count in range(min(cap, len(remaining)), 0, -1):
            rejection = tier_allowed(count, limits, config['resource_limits'], rss, gpu_per_worker,
                                     gpu['memory_total_MiB'])
            if not rejection:
                break
        else:
            raise RuntimeError('Resource headroom is insufficient for a single BC worker: ' + ', '.join(rejection))
        wave, remaining = remaining[:count], remaining[count:]
        index = len(waves) + 1
        label = f'BC_wave_{index:02d}'
        write(base / f'{label}.GPU_ADMISSION.json', idle_gpu(gpu['uuid']))
        names = [job['id'] for job in wave]
        print(stage, label, 'starting', names, flush=True)
        measured = monitor_jobs(base, names, [launch_python('bc.py', 'train', paths[name]) for name in names],
                                [process_env(gpu['uuid']) for _ in names], limits, config, gpu, label)
        completions = [read(Path(job['output']) / 'COMPLETED.json') for job in wave]
        rates = {job['id']: value['steps'] / value['training_seconds'] for job, value in zip(wave, completions)}
        measured.update(index=index, worker_rates=rates, wall_updates_per_second=200000 * count / measured['wall_seconds'],
                        sum_training_updates_per_second=sum(rates.values()))
        write(base / f'{label}.THROUGHPUT.json', measured)
        waves.append(measured)
        if count > 1 and not any(
                (OUTPUTS / name / 'FIRST_BC_CONCURRENCY.json').exists() for name in (STAGE_OUTPUTS['1A'], STAGE_OUTPUTS['1B'])):
            write(base / 'FIRST_BC_CONCURRENCY.json', measured)
            print('FIRST_BC_CONCURRENCY', json.dumps(measured), flush=True)
        rss = max(rss or 0, measured['peak_worker_RSS_bytes'])
        gpu_per_worker = max(gpu_per_worker or 0, measured['peak_gpu_used_MiB'] / count)
        for job in wave:
            name = job['id']
            monitor_jobs(base, [name + '.audit'], [launch_python('bc.py', 'audit', paths[name])],
                         [process_env()], limits, config, gpu, name + '.audit')
        if measured['pressure_warnings'] and remaining:
            raise RuntimeError('Persistent shared resource pressure. Completed/audited jobs were preserved: '
                               + ', '.join(measured['pressure_warnings']))
    return jobs, waves


def root_report(base, resources, phases):
    weight = read(base / 'WEIGHTS_COMPLETED.json')
    training = read(base / 'TRAINING_COMPLETED.json')
    comparison = read(base / 'ROOT_DATA_ENV_COMPARISON.json')
    config = read(base / 'config.json')
    assert read(base / 'AUDIT.json')['passed']
    estimate = training['estimated_1M_seconds_without_weight_computation']
    lines = [
        '# V12 阶段 2 ROOT 迁移试跑报告', '',
        f"上游 commit `{config['commit']}`。会话 `{resources['tmux_session']}`，入口 `{resources['launch']}`。",
        f'输出 `{base}`。独立环境 `{ROOT_PREFIX}`，环境清单见 ENVIRONMENT_BEFORE/AFTER。',
        'Walker2d-morph medium→medium，seed 901。原 VAE 预训练 10,000 步及 ROOT 主循环 20,000 步全部完成。',
        '主训练 60,000 次 Adam 更新，VAE 另有 10,000 次 Adam 更新。两次目标评估共 20 回合。', '',
        *cpu_lines(resources), '', '## 数据与评估核对', '',
        '|任务|HDF5 逐数组|XML 字节|原 ROOT 目标行|V12 显式 next 行|相邻行替代差异／非 terminal 差异|ROOT 归一化|V12 归一化|',
        '|---|---|---|---:|---:|---|---|---|',
    ]
    for task, value in comparison.items():
        lines.append(f"|{task}|{'相同' if value['arrays_equal'] else '不同'}|{'相同' if value['xml_bytes_equal'] else '不同'}|"
                     f"{value['original_ROOT_rows']}|{value['explicit_next_rows']}|{value['shifted_next_rows_differ']} / "
                     f"{value['shifted_next_nonterminal_rows_differ']}|{value['ROOT_score_constants']}|{value['V12_score_constants']}|")
    lines += [
        '', '两端构造 v3 目标环境，XML timestep 0.002，TimeLimit 1,000。ROOT 上游默认每 10,000 步对源域和目标域各评估 10 回合，'
        '输出当前点目标域 10 回合平均原始回报的 0–100 归一化分数。V12 入口在目标域按固定逐回合种子评估并保存长度，'
        '评估前后恢复训练 RNG。同一批回报额外计算上游常数下的分数。',
        'V12 使用迁移包显式 next_observations。源域保留 D4RL v2 的 canonical 行顺序，目标域保留 5,000 行。'
        '这会改变上游相邻行构造的 OT 特征和训练转移，属于 V12 本机协议适配。', '',
        '## 鲁棒 OT 权重', '',
        f"在 {weight['jax_device']} 上调用原求解器一次，处理 {weight['rows']:,} 源行、{weight['source_chunks']} 批。"
        f"计算及写出用时 {weight['seconds']:.2f} 秒。该子进程退出后 PyTorch 子进程读取同一 HDF5，保留原归一化与截断。",
        f"权重阶段采样显存峰值 {phases['weights']['peak_gpu_used_MiB']} MiB，"
        f"子进程峰值 RSS {phases['weights']['peak_worker_RSS_bytes'] / 1024 ** 2:.1f} MiB。",
        f"归一化后超过 5 并实际截断的比例 {weight['fraction_clipped_at_5']:.4%}，原始零权重比例 {weight['fraction_zero']:.4%}。",
        '', '|权重|均值|最小|P01|P25|P50|P75|P95|P99|最大|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for key in ('raw', 'normalized_float32', 'clipped'):
        value = weight[key]
        numbers = [value['mean']] + [value['quantiles'][q] for q in ('min', 'p01', 'p25', 'p50', 'p75', 'p95', 'p99', 'max')]
        lines.append('|' + key + '|' + '|'.join(f'{v:.6g}' for v in numbers) + '|')
    lines += ['', '## 短跑评估与耗时', '', '|步数|V12 目标分数|ROOT 常数分数|回合长度均值|中位数|最短|<200 步比例|',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for step in (10000, 20000):
        event = read(base / f'eval_{step:06d}.json')
        length = event['length_statistics']
        lines.append(f"|{step}|{event['mean']:.4f}|{event['ROOT_author_normalized_score']:.4f}|{length['mean']:.1f}|"
                     f"{length['median']:.1f}|{length['minimum']}|{length['fraction_under_200']:.1%}|")
    lines += [
        '', f"VAE {training['VAE_seconds']:.2f} 秒，主循环 {training['training_seconds']:.2f} 秒，"
        f"两次评估合计 {sum(training['evaluation_seconds']):.2f} 秒。"
        f"单次 100 万步外推 {estimate / 3600:.2f} 小时，含一次 VAE 和估计的 100 次目标评估。"
        f"若另计算一次权重，合计 {((estimate + weight['seconds']) / 3600):.2f} 小时。后期回合可能更长。",
        '两处评估点为迁移试跑记录。Last5 不适用，本阶段完成后进入 AWAITING_STAGE3_CONFIRMATION。', '',
        '## ROOT 配置', '',
        'README 的版本要求为 JAX 0.4.9、jaxlib 0.4.9、ott-jax 0.4.5、jaxopt 0.8.3。完整上游 YAML、实际 CLI 默认值、'
        '维度覆盖、正则和求解器常量均保存在 config.json。上游 YAML 的 metric 为 cosine，run_root.sh 未传 metric，'
        'train_root.py 的默认 euclidean 会覆盖它。原 learner 内部广播、截断、VAE、采样及更新顺序直接沿用固定 commit。',
        '', '```json', json.dumps(config, ensure_ascii=False, indent=2), '```', '',
    ]
    (base / 'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    write(base / 'RESULT.json', dict(stage='2', training=training, weights=weight, data_comparison=comparison,
                                     resource_usage=resources, phases=phases, Last5_applicable=False))


def run_root(base, config, gpu):
    limits = limits_snapshot()
    assert not ROOT_PREFIX.exists(), f'ROOT environment already exists: {ROOT_PREFIX}'
    assert shutil.disk_usage(base).free >= 12 * 1024 ** 3, 'ROOT environment requires 12 GiB free headroom'
    write(base / 'ROOT_ENVIRONMENT_PLAN.json', dict(
        cloned_from=sys.prefix, isolated_prefix=str(ROOT_PREFIX), conda_copy=True,
        JAX_process_separate=True, preallocate_GPU_memory=False, solver_unchanged=True,
        note='The JAX worker computes weights once and exits. The Torch worker reads the same float values and applies the original global mean normalization and training clamp.',
    ))
    phases = {}
    cmd = ['bash', str(HERE / 'root_env.sh'), str(ROOT_PREFIX), str(base), str(Path(sys.prefix))]
    phases['environment'] = monitor_jobs(base, ['root_environment'], [cmd], [process_env()], limits, config, gpu, 'root_environment')
    job = dict(stage='2', id='root_walker2d_morph_m_m_s901', task='walker2d-morph', seed=901,
               controller_pid=os.getpid(), output=str(base), project=str(PROJECT), gpu_uuid=gpu['uuid'])
    job_file = base / 'root_job.json'
    write(job_file, job)
    write(base / 'AUTHORIZATION.json', dict(stage='2', job_sha256=sha(job_file), scope='one OT solve and one 20k ROOT trial',
                                          stage3_authorized=False))
    for mode in ('prepare', 'weights', 'train', 'audit'):
        limits = limits_snapshot()
        is_gpu = mode in ('weights', 'train')
        if is_gpu:
            reason = tier_allowed(1, limits, config['resource_limits'], 2 * 1024 ** 3, None, gpu['memory_total_MiB'])
            assert not reason, reason
            write(base / f'root_{mode}.GPU_ADMISSION.json', idle_gpu(gpu['uuid']))
        command = launch_python('root_trial.py', mode, job_file, ROOT_PREFIX)
        env = process_env(gpu['uuid'] if is_gpu else None, ROOT_PREFIX, jax=mode == 'weights')
        phases[mode] = monitor_jobs(base, ['root_' + mode], [command], [env], limits, config, gpu, 'root_' + mode)
        if is_gpu and phases[mode]['pressure_warnings'] and mode == 'weights':
            raise RuntimeError('Resource pressure during ROOT weights. Weights preserved, training not launched.')
    return phases


def check_inputs(frozen, project_before, env_before):
    assert verify_project(PROJECT, frozen['project_manifest_sha256']) == project_before
    check_environment_protection(env_before)
    verify_package()


def run_stage(stage, frozen, project_before, env_before, config, gpu, session, launch):
    base = OUTPUTS / STAGE_OUTPUTS[stage]
    base.mkdir(exist_ok=False)
    (base / 'logs').mkdir()
    started = time.monotonic()
    limits = limits_snapshot()
    limits['nproc_all'] = int(subprocess.check_output(['nproc', '--all'], text=True).strip())
    host_before, affinity_before = cpu_ticks(), cpu_ticks(limits['affinity'])
    try:
        write(base / 'STARTED.json', dict(stage=stage, session=session, code_freeze_sha256=sha(HERE / 'FROZEN.json'),
                                         GPU=gpu, nvidia_smi=full_smi(), limits=limits, time=utc()))
        if stage == '2':
            phases = run_root(base, config, gpu)
        else:
            jobs, waves = run_bc(stage, base, config, gpu)
        resources = inventory(base, started, limits, host_before, affinity_before, session, launch)
        check_inputs(frozen, project_before, env_before)
        if stage == '2':
            root_report(base, resources, phases)
        else:
            bc_report(stage, base, jobs, resources, waves)
        write(base / 'COMPLETED.json', dict(
            completed=True, stage=stage, input_protection_passed=True,
            workers_audited=1 if stage == '2' else len(jobs),
            report_sha256=sha(base / 'REPORT_ZH.md'), result_sha256=sha(base / 'RESULT.json'),
            status='AWAITING_STAGE3_CONFIRMATION' if stage == '2' else 'DIAGNOSTIC_COMPLETED',
            stage3_started=False, ended_utc=utc(), elapsed_seconds=time.monotonic() - started,
        ))
        print(f'V12 STAGE {stage} COMPLETE: {base / "REPORT_ZH.md"}', flush=True)
    except BaseException:
        error = traceback.format_exc()
        try:
            check_inputs(frozen, project_before, env_before)
            protection = True
        except BaseException:
            protection = False
        write(base / 'FAILED.json', dict(stage=stage, error=error, original_input_protection_passed=protection,
                                        automatic_retry=False, next_stage_started=False, time=utc()))
        raise


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and sys.version_info[:3] == (3, 8, 13) and os.environ.get('TMUX')
    assert Path(sys.prefix).resolve() == (HOME_DIR / 'conda-envs/offdynamics38').resolve()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    lock = os.open(str(HERE), os.O_RDONLY)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        for name in STAGE_OUTPUTS.values():
            assert not (OUTPUTS / name).exists(), f'Output exists. Preserve it: {OUTPUTS / name}'
        assert not ROOT_PREFIX.exists(), f'ROOT environment exists. Preserve it: {ROOT_PREFIX}'
        frozen = verify_package()
        config = read(HERE / 'config_template.json')
        verified = stage0_verified()
        project_before = verify_project(PROJECT, frozen['project_manifest_sha256'])
        check_environment_protection(read(STAGE0 / 'ENVIRONMENT_BEFORE.json'))
        env_before = existing_environment_snapshot()
        gpu = idle_gpu(args.gpu)
        assert gpu['uuid'] == verified['selected_GPU_UUID']
        start_path = OUTPUTS / 'v12_stage1_2_20261107_preflight.json'
        write(start_path, dict(stage0=verified, old_project_manifest_sha256=frozen['project_manifest_sha256'],
                              original_environment=env_before, single_GPU_UUID=gpu['uuid'], maximum_GPU_workers=CAP,
                              authorized_stages=['1A', '1B', '2'], time=utc()))
        session = subprocess.check_output(['tmux', 'display-message', '-p', '#S'], text=True).strip()
        launch = f'bash {HERE / "run_in_tmux.sh"} {args.gpu}'
        for stage in ('1A', '1B', '2'):
            run_stage(stage, frozen, project_before, env_before, config, gpu, session, launch)
        stage0_verified()
        print('AWAITING_STAGE3_CONFIRMATION', flush=True)
    finally:
        os.close(lock)


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'Controller received signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupt)
    main()
