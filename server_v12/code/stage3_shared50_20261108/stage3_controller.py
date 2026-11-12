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

from stage3_setup import BASE_OUT, CAP, GPU_UUIDS, HERE, HOME_DIR, LEGACY, ORIGINAL_OUT, OUT, ROOT_COMMIT, SEEDS, STAGE2, STEPS, TASK
from stage3_setup import load_legacy, read, sha, source_mode, verify_new
from stage3_report import prelaunch_budget, q_lines, summary
from stage3_worker import compare_weight_distributions, root_source_proof, solver_seed_proof
import gpu_admission
import handoff
import recovery


def wait_gpu(label, common, gpu, requested=1, processes=()):
    return gpu_admission.wait([gpu['index']], label, common, gpu['driver'], OUT, requested, processes)


def verify_prior(legacy):
    _, common, controller, _, trial, _ = legacy
    phase0 = controller.stage0_verified()
    assert phase0['selected_GPU_UUID'] == GPU_UUIDS[0]
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
    assert cfg['learner']['metric'] == 'euclidean' and cfg['gpu_uuid'] == GPU_UUIDS[0]
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


def job_entries(controller, common, uuid, reused):
    training, all_jobs = [], {}
    for group, steps in STEPS.items():
        for seed in SEEDS:
            name = f'{group}_s{seed}'
            job = dict(id=name, task=TASK, stage='3', group=group, seed=seed, steps=steps,
                       gpu_uuid=uuid, controller_pid=os.getpid(), project=str(controller.PROJECT),
                       output=str(OUT / 'runs' / name), release=str(OUT / 'gates' / (name + '.json')))
            if name in reused:
                job = reused[name]['job']
            training.append(job)
            all_jobs[name] = job
    for name in ('probe', 'weights'):
        all_jobs[name] = dict(id=name, task=TASK, stage='3', group='ROOT',
                              gpu_uuid=uuid, controller_pid=os.getpid(), project=str(controller.PROJECT),
                              output=str(OUT if name == 'probe' else OUT / 'weights'),
                              release=str(OUT / 'gates' / (name + '.json')))
    paths = {}
    for name, job in all_jobs.items():
        paths[name] = OUT / 'jobs' / (name + '.json')
        common.write(paths[name], job)
    common.write(OUT / 'AUTHORIZATION.json', dict(scope='V12_STAGE3_3_GPU_PROCESSES_ONLY',
                 max_gpu_workers=3, GPU_UUID=uuid, jobs={name: sha(path) for name, path in paths.items()},
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
    uuid = gpu['uuid']
    processes, handles = [], []
    started = time.monotonic()
    monitor = resources.Monitor(limits, policy['resource_limits'], lambda: common.gpu_info(uuid))
    try:
        # All children remain CPU-side at their release markers until all admissions succeed.
        for job in jobs:
            name = job['id']
            waits.append(wait_gpu(f'{label}_{name}', common, gpu, len(jobs), processes))
            runtime = OUT / 'runtime' / name
            for suffix in ('tmp', 'cache', 'cuda'):
                (runtime / suffix).mkdir(parents=True, exist_ok=False)
            prefix = controller.ROOT_PREFIX if job['group'] == 'ROOT' else None
            env = controller.process_env(uuid, prefix=prefix, jax=mode == 'weights')
            env.update(TMPDIR=str(runtime / 'tmp'), XDG_CACHE_HOME=str(runtime / 'cache'),
                       CYTHON_CACHE_DIR=str(runtime / 'cache/cython'), CUDA_CACHE_PATH=str(runtime / 'cuda'))
            stdout = (OUT / 'logs' / (name + '.stdout.log')).open('x', encoding='utf-8')
            stderr = (OUT / 'logs' / (name + '.stderr.log')).open('x', encoding='utf-8')
            handles.extend([stdout, stderr])
            cmd = command(job, mode, paths[name], controller)
            proc = subprocess.Popen(cmd, env=env, cwd=OUT, stdout=stdout, stderr=stderr)
            processes.append(proc)
            common.write(OUT / 'logs' / (name + '.launch.json'), dict(pid=proc.pid, ppid=os.getpid(),
                         command=cmd, job_sha256=sha(paths[name]), GPU_UUID=uuid, time=common.utc()))
        admission = wait_gpu(label + '_release', common, gpu, len(jobs), processes)
        waits.append(admission)
        assert all(p.poll() is None for p in processes)
        assert len(processes) <= CAP
        names = [j['id'] for j in jobs]
        grant = dict(GPU_UUID=uuid, jobs=names, controller_pid=os.getpid(), time=common.utc())
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
                      selected_GPU_UUID=uuid, external_GPU_pids_at_release=admission['pids'],
                      total_GPU_processes_at_release=len(admission['pids']) + len(processes),
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


def prepare_weights(legacy, all_jobs, paths, limits, policy, gpu, waits, reused_weights):
    _, common, _, _, _, _ = legacy
    inputs = read(OUT / 'ROOT_INPUTS.json')
    former = read(STAGE2 / 'ROOT_INPUTS.json')
    for key in ('source_rows', 'source_sha256', 'source_rows_sha256', 'target_rows', 'target_sha256'):
        assert inputs[key] == former[key], key
    proof = read(OUT / 'ROOT_OT_SEED_INDEPENDENCE.json')
    assert proof['passed'] and not proof['training_seed_input'] and not proof['RNG_use_in_pinned_solver']
    if reused_weights is None:
        rejected = legacy[3].tier_allowed(1, limits, policy['resource_limits'], 2 * 1024 ** 3, None, gpu['memory_total_MiB'])
        assert not rejected, rejected
        (OUT / 'weights').mkdir()
        shutil.copyfile(OUT / 'ROOT_INPUTS.json', OUT / 'weights/ROOT_INPUTS.json')
        record = run_gpu_wave('ROOT_weights', [all_jobs['weights']], 'weights', paths, legacy, limits, policy, gpu, waits)
        receipt = read(OUT / 'weights/WEIGHTS_COMPLETED.json')
        path = OUT / 'weights/robust_weights.hdf5'
        assert not record['pressure_warnings'], 'Resource pressure during ROOT weights. Weights preserved.'
    else:
        assert reused_weights['inputs'] == inputs
        receipt = reused_weights['receipt']
        path = Path(reused_weights['path'])
    assert receipt['completed'] and receipt['sha256'] == sha(path)
    assert receipt['CUDA_VISIBLE_DEVICES'] == gpu['uuid']
    assert receipt['source_rows_sha256'] == inputs['source_rows_sha256'] and receipt['rows'] == inputs['source_rows']
    comparison = compare_weight_distributions(path, STAGE2 / 'robust_weights.hdf5', receipt, read(STAGE2 / 'WEIGHTS_COMPLETED.json'))
    common.write(OUT / 'ROOT_WEIGHT_COMPARISON.json', comparison)
    origin = 'one_upstream_adjacent_source_solve_for_all_seeds' if reused_weights is None else 'verified_completed_stage3_weights_reused'
    common.write(OUT / 'ROOT_TRAINING_WEIGHTS.json', dict(receipt, origin=origin,
                 path=str(path), source_mode=source_mode(), formal_training_seeds=list(SEEDS),
                 seed_independence_sha256=sha(OUT / 'ROOT_OT_SEED_INDEPENDENCE.json'),
                 comparison_sha256=sha(OUT / 'ROOT_WEIGHT_COMPARISON.json')))


def reuse_cpu_probe(legacy, history):
    names = ('ROOT_Q_READONLY_PROBE.json', 'ROOT_INPUTS.json', 'ROOT_OT_SEED_INDEPENDENCE.json')
    source = next((d for d in reversed(history) if all((d / name).is_file() for name in names)), None)
    if source is None:
        return False
    _, common, _, _, trial, _ = legacy
    probe = read(source / names[0])
    assert probe['passed'] and probe['CPU_only'] and not probe['CUDA_initialized']
    assert probe['optimizer_updates'] == probe['environment_steps'] == 0
    assert probe['checkpoint_sha256'] == sha(STAGE2 / 'final_checkpoint.pt')
    assert probe['weight_sha256'] == sha(STAGE2 / 'robust_weights.hdf5')
    assert all(probe[key] == value for key, value in root_source_proof(trial).items())
    assert read(source / names[2]) == solver_seed_proof(trial)
    inputs = read(source / names[1])
    assert inputs['ROOT_source_next'] == source_mode() and inputs['target_explicit_next']
    assert inputs['source_next_different_rows'] == 513 and inputs['target_next_different_rows'] == 7
    assert sha(inputs['source']) == inputs['source_sha256'] and sha(inputs['target']) == inputs['target_sha256']
    for name in names:
        shutil.copyfile(source / name, OUT / name)
    common.write(OUT / 'CPU_PROBE_REUSED.json', dict(passed=True, source=str(source), artifacts={name: sha(OUT / name) for name in names}))
    return True


def run(legacy, frozen, policy, initial, registered, transfer, history, reused, interrupted, weights, resume):
    _, common, controller, resources, _, _ = legacy
    assert not OUT.exists(), 'Stage 3 output exists; preserve previous results'
    prior = verify_prior(legacy)
    project_before = common.verify_project(controller.PROJECT, frozen['project_manifest_sha256'])
    common.check_environment_protection(read(controller.STAGE0 / 'ENVIRONMENT_BEFORE.json'))
    old_env = common.existing_environment_snapshot()
    gpu = initial['gpu']
    original_hashes = {path: common.hash_tree(path) for path in history}
    OUT.mkdir(exist_ok=False)
    for name in ('jobs', 'logs', 'gates', 'admissions', 'runs', 'reuse_audits'):
        (OUT / name).mkdir()
    start = time.monotonic()
    limits = resources.limits_snapshot()
    limits['nproc_all'] = int(subprocess.check_output(['nproc', '--all'], text=True).strip())
    host_before = resources.cpu_ticks()
    affinity_before = resources.cpu_ticks(limits['affinity'])
    session = subprocess.check_output(['tmux', 'display-message', '-p', '#S'], text=True).strip()
    launch = f'bash {HERE / "run_in_tmux.sh"} --registered ' + ' '.join(map(str, registered))
    if resume:
        launch += ' --resume'
    waves, waits = [], [initial]

    def check_old():
        controller.check_inputs(frozen, project_before, old_env)
        assert verify_prior(legacy) == prior
        assert all(common.hash_tree(path) == hashes for path, hashes in original_hashes.items())
        recovery.verify_history_code(history)

    try:
        common.write(OUT / 'STARTED.json', dict(time=common.utc(), GPU=gpu, tmux=session, max_GPU_workers=CAP,
                     registered_GPU_indices=registered, user_confirmed_registered=True,
                     controller_pid=os.getpid(), nvidia_smi=common.full_smi(), limits=limits))
        common.write(OUT / 'admissions/initial_selection.json', initial)
        common.write(OUT / 'ORIGINAL_QUEUE_HANDOFF.json', dict(transfer, original_files_protected=sum(map(len, original_hashes.values()))))
        common.write(OUT / 'PRIOR_VERIFIED.json', prior)
        shutil.copyfile(HERE / 'PROTOCOL.json', OUT / 'PROTOCOL.json')
        jobs, all_jobs, paths = job_entries(controller, common, gpu['uuid'], reused)
        remaining = [job for job in jobs if job['id'] not in reused]
        restart_plan = dict(resumed=resume, attempt_path=str(OUT), previous_outputs=list(map(str, history)),
                            reused_jobs=reused, previous_interrupted_runs=interrupted,
                            newly_scheduled_jobs=[job['id'] for job in remaining],
                            verified_weights_path=weights['path'] if weights is not None else None,
                            restart_policy='Interrupted seeds restart from step 0 in this attempt; no in-seed restore')
        common.write(OUT / 'RESTART_PLAN.json', restart_plan)
        budget = prelaunch_budget(OUT, common, initial, [job['group'] for job in remaining], weights is None)
        print((OUT / 'PRELAUNCH_REPORT_ZH.md').read_text(encoding='utf-8'), flush=True)
        common.write(OUT / 'PRE_RESULT_COMMITMENT.json', dict(before_ANY_GPU_workers=True, time=common.utc(),
                     scope='stage3 only', CAP=3, protocol_sha256=sha(OUT / 'PROTOCOL.json'),
                     restart_plan_sha256=sha(OUT / 'RESTART_PLAN.json'),
                     authorization_sha256=sha(OUT / 'AUTHORIZATION.json'), budget_sha256=sha(OUT / 'BUDGET_ESTIMATE.json')))
        if not reuse_cpu_probe(legacy, history):
            run_cpu(all_jobs['probe'], 'probe', paths['probe'], legacy, limits, policy, gpu)
        probe = read(OUT / 'ROOT_Q_READONLY_PROBE.json')
        assert probe['passed'] and probe['optimizer_updates'] == 0 and probe['environment_steps'] == 0
        q_text = '# V12 阶段 3 CPU 核验\n\n' + '\n'.join(q_lines(probe))
        with (OUT / 'ROOT_Q_READONLY_ZH.md').open('x', encoding='utf-8') as file:
            file.write(q_text)
        print(q_text, flush=True)
        prepare_weights(legacy, all_jobs, paths, resources.limits_snapshot(), policy, gpu, waits, weights)
        audit_paths = {}
        current_weights = read(OUT / 'ROOT_TRAINING_WEIGHTS.json')
        for job in jobs:
            name = job['id']
            if name not in reused:
                continue
            if job['group'] == 'ROOT':
                assert read(Path(job['output']) / 'config.json')['weight_sha256'] == current_weights['sha256']
            if reused[name]['audit'] is None:
                run_cpu(job, 'reaudit', paths[name], legacy, limits, policy, gpu)
                receipt = OUT / 'reuse_audits' / (name + '.json')
            else:
                receipt = Path(reused[name]['audit'])
            recovery.verify_audit(job, receipt)
            audit_paths[name] = receipt
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
            label = 'wave_%02d' % (len(waves) + 1)
            slots = wait_gpu(label + '_capacity', common, gpu)
            waits.append(slots)
            n = min(n, slots['slots'])
            wave, remaining = remaining[:n], remaining[n:]
            for job in wave:
                Path(job['output']).mkdir(exist_ok=False)
            measured = run_gpu_wave(label, wave, 'train', paths, legacy, current, policy, gpu, waits)
            waves.append(measured)
            rss = max(rss, measured['peak_worker_RSS_bytes'])
            for job in wave:
                run_cpu(job, 'audit', paths[job['id']], legacy, current, policy, gpu)
                receipt = Path(job['output']) / 'AUDIT.json'
                assert read(receipt)['passed']
                audit_paths[job['id']] = receipt
            if measured['pressure_warnings']:
                cap = 1
                print('Resource pressure observed; subsequent waves limited to one worker', flush=True)
        max_own = max((wave['workers'] for wave in waves), default=0)
        assert sum(wave['workers'] for wave in waves) + len(reused) == len(audit_paths) == 15 and max_own <= CAP
        overall = controller.inventory(OUT, start, limits, host_before, affinity_before, session, launch)
        check_old()
        common.write(OUT / 'INPUT_PROTECTION_AFTER.json', dict(passed=True, transfer_files_rechecked=len(project_before),
                     transfer_manifest_sha256=frozen['project_manifest_sha256'], legacy_freeze_sha256=sha(LEGACY / 'FROZEN.json'),
                     stage3_freeze_sha256=sha(HERE / 'FROZEN.json'), prior_verified_sha256=sha(OUT / 'PRIOR_VERIFIED.json'),
                     original_offdynamics_environment_checked=True, ROOT_environment_pip_freeze_checked=True))
        result = summary(OUT, jobs, legacy, overall, waves, budget, waits, audit_paths)
        common.publish(OUT / 'STAGE3_COMPLETED.json', dict(stage='3', completed=True, CPU_audited_jobs=15, own_GPU_workers_exited=True,
                     original_inputs_protected=True, max_observed_own_training_workers_this_attempt=max_own,
                     reused_completed_jobs=len(reused), new_jobs=15 - len(reused), restart_plan_sha256=sha(OUT / 'RESTART_PLAN.json'),
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
    global OUT
    import fcntl
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--registered', type=int, choices=tuple(GPU_UUIDS), nargs='+')
    choice.add_argument('--scan', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert not (args.scan and args.resume)
    assert sys.platform == 'linux' and sys.version_info[:3] == (3, 8, 13)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1' and Path(sys.prefix).resolve() == (HOME_DIR / 'conda-envs/offdynamics38').resolve()
    verify_new()
    legacy = load_legacy()
    frozen = legacy[1].verify_package()
    _, common, controller, _, _, _ = legacy
    policy = read(LEGACY / 'config_template.json')
    driver = policy['resource_admission']['observed_driver']
    if args.scan:
        ordered, rows = gpu_admission.candidates(GPU_UUIDS, common, driver)
        for row in rows:
            print(f"GPU {row['index']}: {row['reason']}; admissible jobs={row['slots']}; visible PIDs={row['pids']}")
        print('Preferred order:', ', '.join(str(row['index']) for row in ordered) or 'none')
        return
    assert os.environ.get('TMUX'), 'Run the queue inside tmux'
    registered = sorted(set(args.registered))
    if not args.resume:
        def fresh_guard():
            assert all(path == ORIGINAL_OUT for path in recovery.history()), 'An earlier queue exists; use resume after it exits'
            allowed = tuple(os.fsencode(handoff.ORIGINAL_CODE / name) for name in ('stage3_controller.py', 'stage3_worker.py'))
            recovery.assert_no_live(allowed)
            handoff.require_no_release()

        initial = gpu_admission.wait(registered, 'initial_selection', common, driver, before_poll=fresh_guard)
        print(f"Selected GPU {initial['index']} ({initial['reason']}); checking original queue", flush=True)
        transfer = handoff.stop_waiting_original()
    lock = os.open(str(LEGACY), os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovery.assert_no_live()
        past = recovery.history()
        recovery.verify_history_code(past)
        if args.resume:
            assert past, 'No earlier stage 3 output to resume; use the normal launch'
            uuid = recovery.released_gpu(past)
            reused, interrupted = recovery.collect(past, controller.PROJECT, uuid)
            weights = recovery.completed_weights(past)
            if any(list((base / 'gates').glob('ROOT_s*.json')) for base in past):
                assert weights is not None, 'ROOT training was released but its completed weights are missing'
            candidates = registered
            if uuid is not None:
                index = next(i for i, value in GPU_UUIDS.items() if value == uuid)
                assert index in registered, f'Register the same physical GPU {index} to resume this stage'
                candidates = [index]
            print(f'Restart plan: {len(reused)} fully trained, {15 - len(reused)} remaining, {len(interrupted)} interrupted attempts preserved', flush=True)
            if len(reused) == 15:
                assert weights is not None and uuid is not None
                gpu = common.gpu_info(uuid)
                assert gpu['uuid'] == uuid and gpu['driver'] == driver
                initial = dict(gpu=gpu, index=gpu['index'], slots=0, blocked_wait_seconds=0., reason='CPU-only report and audit recovery')
            else:
                initial = gpu_admission.wait(candidates, 'resume_selection', common, driver)
            transfer = dict(resume_no_live_processes_verified=True, old_GPU_released=uuid is not None, signalled=False)
            number, OUT = recovery.next_attempt()
        else:
            assert all(path == ORIGINAL_OUT for path in past)
            handoff.require_no_release()
            reused, interrupted, weights = {}, [], None
            number, OUT = 1, BASE_OUT
        os.environ['V12_STAGE3_ATTEMPT'] = str(number)
        print('Writing attempt:', OUT, flush=True)
        run(legacy, frozen, policy, initial, registered, transfer, past, reused, interrupted, weights, args.resume)
    finally:
        os.close(lock)


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'Controller received signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupt)
    main()
