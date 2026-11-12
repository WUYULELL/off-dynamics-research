"""V12 stage 3 queue. One registered GPU, at most three GPU workers."""
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

from stage3_setup import CAP, GPU_UUID, HERE, HOME_DIR, LEGACY, OUT, ROOT_COMMIT, SEEDS, STAGE2, STEPS, TASK
from stage3_setup import load_legacy, read, sha, source_mode, verify_new
from stage3_report import prelaunch_budget, q_lines, summary
from stage3_worker import compare_weight_distributions


WAITABLE = {
    'Memory exceeds the diagnosed resident-memory allowance',
    'GPU utilization is nonzero',
    'An active process is visible on the selected GPU: selected_compute_processes',
    'An active process is visible on the selected GPU: selected_pmon_processes',
}


def occupancy_failure(exc):
    if not exc.args or not isinstance(exc.args[0], tuple) or len(exc.args[0]) != 2:
        return None
    message, samples = exc.args[0]
    if message != 'Selected GPU admission blocked; no retry/switch/signals' or not samples:
        return None
    reasons = set(samples[-1]['rejection_reasons'])
    return samples[-1] if reasons and reasons <= WAITABLE else None


def wait_gpu(out, label, common, processes=()):
    started = time.monotonic()
    blocked_since = None
    polls = 0
    while True:
        assert all(proc.poll() is None for proc in processes), 'A waiting child exited before GPU release'
        try:
            result = common.idle_gpu(GPU_UUID)
        except AssertionError as exc:
            busy = occupancy_failure(exc)
            if busy is None:
                raise
            if blocked_since is None:
                blocked_since = time.monotonic()
            polls += 1
            event = dict(label=label, time=common.utc(), wait_seconds=time.monotonic() - blocked_since,
                         selected=busy['gpu'], reasons=busy['rejection_reasons'],
                         selected_compute_processes=busy['selected_compute_processes'], selected_pmon_processes=busy['selected_pmon_processes'])
            with (out / 'GPU_WAIT.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(event, allow_nan=False) + '\n')
            print(label, 'GPU occupied; waiting 60 seconds', event['reasons'], flush=True)
            time.sleep(60)
        else:
            finished = time.monotonic()
            info = dict(label=label, blocked_polls=polls, elapsed_seconds=finished - started,
                        blocked_wait_seconds=0. if blocked_since is None else finished - blocked_since,
                        checked_utc=common.utc(), gpu=result)
            common.write(out / 'admissions' / (label + '.json'), info)
            return info


def verify_prior(legacy):
    _, common, controller, _, trial, _ = legacy
    phase0 = controller.stage0_verified()
    assert phase0['selected_GPU_UUID'] == GPU_UUID
    bc = {}
    for stage, count in (('1A', 5), ('1B', 30)):
        d = controller.OUTPUTS / controller.STAGE_OUTPUTS[stage]
        v = read(d / 'COMPLETED.json')
        assert v['completed'] and v['workers_audited'] == count and v['stage'] == stage
        assert sha(d / 'REPORT_ZH.md') == v['report_sha256'] and sha(d / 'RESULT.json') == v['result_sha256']
        bc[stage] = sha(d / 'COMPLETED.json')
    stage2 = read(STAGE2 / 'COMPLETED.json')
    assert stage2['completed'] and stage2['stage'] == '2' and stage2['workers_audited'] == 1
    assert stage2['original_input_protection_passed'] and stage2['own_GPU_workers_exited']
    assert not stage2['stage3_started'] and stage2['status'] == 'AWAITING_STAGE3_CONFIRMATION'
    assert sha(STAGE2 / 'REPORT_ZH.md') == stage2['report_sha256'] and sha(STAGE2 / 'RESULT.json') == stage2['result_sha256']
    assert stage2['report_sha256'] == '95bc09ebe8e9721e4bcbd5f8d7faa51a864ccb7060da4531bd6568fc8d20289b'
    ck = read(STAGE2 / 'TRAINING_COMPLETED.json')
    weights = read(STAGE2 / 'WEIGHTS_COMPLETED.json')
    audit = read(STAGE2 / 'AUDIT.json')
    cfg = read(STAGE2 / 'config.json')
    assert audit['passed'] and audit['main_updates'] == ck['main_steps'] == 20000 and audit['VAE_updates'] == 10000
    assert sha(STAGE2 / 'final_checkpoint.pt') == ck['checkpoint_sha256'] == audit['checkpoint_sha256']
    assert sha(STAGE2 / 'robust_weights.hdf5') == ck['weight_sha256'] == weights['sha256'] == audit['weight_sha256']
    assert cfg['commit'] == ROOT_COMMIT and cfg['learner']['max_step'] == 1000000
    assert cfg['learner']['metric'] == 'euclidean' and cfg['gpu_uuid'] == GPU_UUID
    assert cfg['solver']['epsilon'] == trial.CLI['epsilon'] == .01
    assert cfg['solver']['lambda_src'] == trial.CLI['lambda_src'] == .05
    assert cfg['solver']['lambda_tar'] == trial.CLI['lambda_tar'] == .5
    assert read(STAGE2 / 'ROOT_DATA_ENV_COMPARISON.json')[TASK]['shifted_next_rows_differ'] == 7
    ready = read(STAGE2 / 'ENVIRONMENT_READY.json')
    assert ready['passed'] and Path(ready['prefix']).resolve() == controller.ROOT_PREFIX.resolve()
    pip = [str(controller.ROOT_PREFIX / 'bin/python'), '-B', '-m', 'pip', 'freeze', '--all']
    installed = subprocess.check_output(pip, env=controller.process_env(prefix=controller.ROOT_PREFIX), encoding='utf-8').splitlines()
    assert installed == (STAGE2 / 'PIP_ENVIRONMENT_AFTER.txt').read_text(encoding='utf-8').splitlines()
    return dict(stage0=phase0, bc_diagnostic_completion_hashes=bc, stage2_completion_hash=sha(STAGE2 / 'COMPLETED.json'),
                stage2_original_hashes={key: sha(STAGE2 / key) for key in (
                    'config.json', 'AUDIT.json', 'REPORT_ZH.md', 'RESULT.json', 'final_checkpoint.pt', 'robust_weights.hdf5',
                    'ROOT_INPUTS.json', 'ROOT_DATA_ENV_COMPARISON.json', 'ENVIRONMENT_READY.json', 'PIP_ENVIRONMENT_AFTER.txt')},
                stage2_environment_versions=ready['versions'])


def job_entries(controller, common):
    training, all_jobs = [], {}
    for group, steps in STEPS.items():
        for seed in SEEDS:
            name = f'{group}_s{seed}'
            job = dict(id=name, task=TASK, stage='3', group=group, seed=seed, steps=steps,
                       gpu_uuid=GPU_UUID, controller_pid=os.getpid(), project=str(controller.PROJECT),
                       output=str(OUT / 'runs' / name), release=str(OUT / 'gates' / (name + '.json')))
            training.append(job)
            all_jobs[name] = job
    for name in ('probe', 'weights'):
        all_jobs[name] = dict(id=name, task=TASK, stage='3', group='ROOT',
                              gpu_uuid=GPU_UUID, controller_pid=os.getpid(), project=str(controller.PROJECT),
                              output=str(OUT if name == 'probe' else OUT / 'weights'),
                              release=str(OUT / 'gates' / (name + '.json')))
    paths = {}
    for name, job in all_jobs.items():
        paths[name] = OUT / 'jobs' / (name + '.json')
        common.write(paths[name], job)
    common.write(OUT / 'AUTHORIZATION.json', dict(scope='V12_STAGE3_3_GPU_PROCESSES_ONLY',
                 max_gpu_workers=3, jobs={name: sha(path) for name, path in paths.items()},
                 training_job_count=15, allowed_groups=list(STEPS), later_stages_authorized=False))
    return training, all_jobs, paths


def command(job, mode, path, controller):
    python = controller.ROOT_PREFIX / 'bin/python' if job['group'] == 'ROOT' else Path(sys.executable)
    return [str(python), '-u', '-B', str(HERE / 'stage3_worker.py'), mode, str(path)]


def run_cpu(job, mode, path, legacy, limits, policy, gpu):
    _, _, controller, _, _, _ = legacy
    name = job['id'] + '.' + mode
    prefix = controller.ROOT_PREFIX if job['group'] == 'ROOT' else None
    return controller.monitor_jobs(OUT, [name], [command(job, mode, path, controller)],
                                   [controller.process_env(prefix=prefix)], limits, policy, gpu, name)


def run_gpu_wave(label, jobs, mode, paths, legacy, limits, policy, gpu, waits):
    _, common, controller, resources, _, _ = legacy
    assert 1 <= len(jobs) <= CAP and mode in ('weights', 'train')
    processes, handles = [], []
    started = time.monotonic()
    monitor = resources.Monitor(limits, policy['resource_limits'], lambda: common.gpu_info(GPU_UUID))
    try:
        # All children remain CPU-side at their release markers until all admissions succeed.
        for job in jobs:
            name = job['id']
            waits.append(wait_gpu(OUT, f'{label}_{name}', common, processes))
            runtime = OUT / 'runtime' / name
            for suffix in ('tmp', 'cache', 'cuda'):
                (runtime / suffix).mkdir(parents=True, exist_ok=False)
            prefix = controller.ROOT_PREFIX if job['group'] == 'ROOT' else None
            env = controller.process_env(GPU_UUID, prefix=prefix, jax=mode == 'weights')
            env.update(TMPDIR=str(runtime / 'tmp'), XDG_CACHE_HOME=str(runtime / 'cache'),
                       CYTHON_CACHE_DIR=str(runtime / 'cache/cython'), CUDA_CACHE_PATH=str(runtime / 'cuda'))
            stdout = (OUT / 'logs' / (name + '.stdout.log')).open('x', encoding='utf-8')
            stderr = (OUT / 'logs' / (name + '.stderr.log')).open('x', encoding='utf-8')
            handles.extend([stdout, stderr])
            cmd = command(job, mode, paths[name], controller)
            proc = subprocess.Popen(cmd, env=env, cwd=OUT, stdout=stdout, stderr=stderr)
            processes.append(proc)
            common.write(OUT / 'logs' / (name + '.launch.json'), dict(pid=proc.pid, ppid=os.getpid(),
                         command=cmd, job_sha256=sha(paths[name]), GPU_UUID=GPU_UUID, time=common.utc()))
        waits.append(wait_gpu(OUT, label + '_release', common, processes))
        assert all(p.poll() is None for p in processes)
        assert len(processes) <= CAP
        names = [j['id'] for j in jobs]
        grant = dict(GPU_UUID=GPU_UUID, jobs=names, controller_pid=os.getpid(), time=common.utc())
        release = time.monotonic()
        for job in jobs:
            common.publish(Path(job['release']), grant)
        print(label, 'GPU started', names, 'PIDs', [p.pid for p in processes], flush=True)
        previous_print = release
        with (OUT / 'logs' / (label + '.resource.jsonl')).open('x', encoding='utf-8') as file:
            while True:
                statuses = [p.poll() for p in processes]
                sample = monitor.sample(processes, label)
                file.write(json.dumps(sample, allow_nan=False) + '\n')
                file.flush()
                if all(status is not None for status in statuses):
                    break
                now = time.monotonic()
                if now - previous_print >= 60:
                    print(label, 'active', [p.pid for p in processes if p.poll() is None], 'elapsed minutes',
                          round((now - release) / 60, 1), flush=True)
                    previous_print = now
                time.sleep(policy['monitor_interval_seconds'])
        # Allow the other already-running jobs to finish if a sibling fails; never refill this wave.
        failures = [(job['id'], rc) for job, rc in zip(jobs, statuses) if rc != 0]
        assert not failures, f'Worker failure in {label}: {failures}. Preserve logs and outputs.'
        record = dict(label=label, jobs=names, workers=len(processes), wall_seconds=time.monotonic() - started,
                      GPU_phase_seconds=time.monotonic() - release, GPU_workers_exited=True,
                      peak_worker_RSS_bytes=monitor.max_worker_RSS,
                      peak_GPU_memory_MiB=max(x['gpu']['memory_used_MiB'] for x in monitor.observed),
                      pressure_warnings=monitor.warnings)
        common.write(OUT / (label + '.COMPLETED.json'), record)
        return record
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for handle in handles:
            handle.close()


def prepare_weights(legacy, all_jobs, paths, limits, policy, gpu, waits):
    _, common, _, _, _, _ = legacy
    inputs = read(OUT / 'ROOT_INPUTS.json')
    former = read(STAGE2 / 'ROOT_INPUTS.json')
    for key in ('source_rows', 'source_sha256', 'source_rows_sha256', 'target_rows', 'target_sha256'):
        assert inputs[key] == former[key], key
    proof = read(OUT / 'ROOT_OT_SEED_INDEPENDENCE.json')
    assert proof['passed'] and not proof['training_seed_input'] and not proof['RNG_use_in_pinned_solver']
    rejected = legacy[3].tier_allowed(1, limits, policy['resource_limits'], 2 * 1024 ** 3, None, gpu['memory_total_MiB'])
    assert not rejected, rejected
    (OUT / 'weights').mkdir()
    shutil.copyfile(OUT / 'ROOT_INPUTS.json', OUT / 'weights/ROOT_INPUTS.json')
    record = run_gpu_wave('ROOT_weights', [all_jobs['weights']], 'weights', paths, legacy, limits, policy, gpu, waits)
    receipt = read(OUT / 'weights/WEIGHTS_COMPLETED.json')
    path = OUT / 'weights/robust_weights.hdf5'
    assert not record['pressure_warnings'], 'Resource pressure during ROOT weights. Weights preserved.'
    assert receipt['completed'] and receipt['sha256'] == sha(path)
    assert receipt['source_rows_sha256'] == inputs['source_rows_sha256'] and receipt['rows'] == inputs['source_rows']
    comparison = compare_weight_distributions(path, STAGE2 / 'robust_weights.hdf5', receipt, read(STAGE2 / 'WEIGHTS_COMPLETED.json'))
    common.write(OUT / 'ROOT_WEIGHT_COMPARISON.json', comparison)
    common.write(OUT / 'ROOT_TRAINING_WEIGHTS.json', dict(receipt, origin='one_upstream_adjacent_source_solve_for_all_seeds',
                 path=str(path), source_mode=source_mode(), formal_training_seeds=list(SEEDS),
                 seed_independence_sha256=sha(OUT / 'ROOT_OT_SEED_INDEPENDENCE.json'),
                 comparison_sha256=sha(OUT / 'ROOT_WEIGHT_COMPARISON.json')))


def run(legacy, frozen):
    _, common, controller, resources, _, _ = legacy
    assert not OUT.exists(), 'Stage 3 output exists; preserve previous results'
    prior = verify_prior(legacy)
    policy = read(LEGACY / 'config_template.json')
    project_before = common.verify_project(controller.PROJECT, frozen['project_manifest_sha256'])
    common.check_environment_protection(read(controller.STAGE0 / 'ENVIRONMENT_BEFORE.json'))
    old_env = common.existing_environment_snapshot()
    gpu = common.gpu_info(GPU_UUID)
    assert gpu['uuid'] == GPU_UUID and gpu['driver'] == policy['resource_admission']['observed_driver']
    OUT.mkdir(exist_ok=False)
    for name in ('jobs', 'logs', 'gates', 'admissions', 'runs'):
        (OUT / name).mkdir()
    start = time.monotonic()
    limits = resources.limits_snapshot()
    limits['nproc_all'] = int(subprocess.check_output(['nproc', '--all'], text=True).strip())
    host_before = resources.cpu_ticks()
    affinity_before = resources.cpu_ticks(limits['affinity'])
    session = subprocess.check_output(['tmux', 'display-message', '-p', '#S'], text=True).strip()
    launch = f'bash {HERE / "run_in_tmux.sh"} --registered'
    waves, waits = [], []

    def check_old():
        controller.check_inputs(frozen, project_before, old_env)
        assert verify_prior(legacy) == prior
        verify_new()

    try:
        common.write(OUT / 'STARTED.json', dict(time=common.utc(), GPU=gpu, tmux=session, max_GPU_workers=CAP,
                     user_confirmed_registered=True, controller_pid=os.getpid(), nvidia_smi=common.full_smi(), limits=limits))
        common.write(OUT / 'PRIOR_VERIFIED.json', prior)
        shutil.copyfile(HERE / 'PROTOCOL.json', OUT / 'PROTOCOL.json')
        budget = prelaunch_budget(OUT, common)
        print((OUT / 'PRELAUNCH_REPORT_ZH.md').read_text(encoding='utf-8'), flush=True)
        jobs, all_jobs, paths = job_entries(controller, common)
        common.write(OUT / 'PRE_RESULT_COMMITMENT.json', dict(before_ANY_GPU_workers=True, time=common.utc(),
                     scope='stage3 only', CAP=3, protocol_sha256=sha(OUT / 'PROTOCOL.json'),
                     authorization_sha256=sha(OUT / 'AUTHORIZATION.json'), budget_sha256=sha(OUT / 'BUDGET_ESTIMATE.json')))
        run_cpu(all_jobs['probe'], 'probe', paths['probe'], legacy, limits, policy, gpu)
        probe = read(OUT / 'ROOT_Q_READONLY_PROBE.json')
        assert probe['passed'] and probe['optimizer_updates'] == 0 and probe['environment_steps'] == 0
        q_text = '# V12 阶段 3 CPU 核验\n\n' + '\n'.join(q_lines(probe))
        with (OUT / 'ROOT_Q_READONLY_ZH.md').open('x', encoding='utf-8') as file:
            file.write(q_text)
        print(q_text, flush=True)
        prepare_weights(legacy, all_jobs, paths, resources.limits_snapshot(), policy, gpu, waits)
        remaining = list(jobs)
        cap = CAP
        rss = 2 * 1024 ** 3
        while remaining:
            current = resources.limits_snapshot()
            for n in range(min(cap, len(remaining)), 0, -1):
                rejected = resources.tier_allowed(n, current, policy['resource_limits'], rss, None, gpu['memory_total_MiB'])
                if not rejected:
                    break
            else:
                raise RuntimeError('Resource checks rejected even one training process: ' + ', '.join(rejected))
            wave, remaining = remaining[:n], remaining[n:]
            label = 'wave_%02d' % (len(waves) + 1)
            for job in wave:
                Path(job['output']).mkdir(exist_ok=False)
            measured = run_gpu_wave(label, wave, 'train', paths, legacy, current, policy, gpu, waits)
            waves.append(measured)
            rss = max(rss, measured['peak_worker_RSS_bytes'])
            for job in wave:
                run_cpu(job, 'audit', paths[job['id']], legacy, current, policy, gpu)
                assert read(Path(job['output']) / 'AUDIT.json')['passed']
            if measured['pressure_warnings']:
                cap = 1
                print('Resource pressure observed; subsequent waves limited to one worker', flush=True)
        assert sum(wave['workers'] for wave in waves) == 15 and max(wave['workers'] for wave in waves) <= CAP
        overall = controller.inventory(OUT, start, limits, host_before, affinity_before, session, launch)
        check_old()
        common.write(OUT / 'INPUT_PROTECTION_AFTER.json', dict(passed=True, transfer_files_rechecked=len(project_before),
                     transfer_manifest_sha256=frozen['project_manifest_sha256'], legacy_freeze_sha256=sha(LEGACY / 'FROZEN.json'),
                     stage3_freeze_sha256=sha(HERE / 'FROZEN.json'), prior_verified_sha256=sha(OUT / 'PRIOR_VERIFIED.json'),
                     original_offdynamics_environment_checked=True, ROOT_environment_pip_freeze_checked=True))
        result = summary(OUT, jobs, legacy, overall, waves, budget, waits)
        common.write(OUT / 'STAGE3_COMPLETED.json', dict(stage='3', completed=True, CPU_audited_jobs=15, own_GPU_workers_exited=True,
                     original_inputs_protected=True, max_observed_own_GPU_workers=max(wave['workers'] for wave in waves),
                     H_root=result['H_root'], baseline=result['selected_baseline'],
                     report_sha256=sha(OUT / 'STAGE3_REPORT_ZH.md'), result_sha256=sha(OUT / 'STAGE3_RESULT.json'),
                     input_protection_sha256=sha(OUT / 'INPUT_PROTECTION_AFTER.json'),
                     elapsed_seconds=time.monotonic() - start, GPU_wait_seconds=result['total_gpu_wait_seconds'],
                     status='AWAITING_USER_DECISION', later_stage_started=False, time=common.utc()))
        print('V12 STAGE 3 COMPLETE:', OUT / 'STAGE3_REPORT_ZH.md', flush=True)
        print('AWAITING_USER_DECISION. Check your program exited and remove your own GPU registration entry.', flush=True)
    except BaseException:
        error = traceback.format_exc()
        try:
            check_old()
            protection = True
        except Exception:
            protection = False
        common.write(OUT / 'FAILED.json', dict(time=common.utc(), error=error, completed_GPU_waves=waves,
                     original_input_protection_passed=protection,
                     recorded_gpu_waits=waits, automatic_retry=False, later_stage_started=False))
        raise


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--registered', action='store_true', required=True)
    parser.parse_args()
    assert sys.platform == 'linux' and sys.version_info[:3] == (3, 8, 13) and os.environ.get('TMUX')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1' and Path(sys.prefix).resolve() == (HOME_DIR / 'conda-envs/offdynamics38').resolve()
    verify_new()
    legacy = load_legacy()
    frozen = legacy[1].verify_package()
    lock = os.open(str(LEGACY), os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(legacy, frozen)
    finally:
        os.close(lock)


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'Controller received signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupt)
    main()
