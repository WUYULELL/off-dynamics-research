"""Continue V12 stage 2 after the recorded pre-GPU admission failure."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback


ORIGINAL_CODE = Path.home() / 'v12_linux/stage1_2_20261107'
ORIGINAL_FREEZE_SHA256 = 'ff6bdb1e4014bd1fea176a25d5040dca8f82b0149d34e9767427d99a04d4ce2d'
ORIGINAL_OUTPUT = Path.home() / 'v12_linux/outputs/stage2_root_20261107'
OUTPUT = Path.home() / 'v12_linux/outputs/stage2_root_20261108_continue1'
PREPARED = ('config.json', 'ROOT_INPUTS.json', 'ROOT_DATA_ENV_COMPARISON.json')
ENVIRONMENT = ('ENVIRONMENT_READY.json', 'ROOT_ENVIRONMENT_PLAN.json', 'CONDA_ORIGINAL.json',
               'PIP_ORIGINAL.txt', 'CONDA_ENVIRONMENT_BEFORE.json', 'PIP_ENVIRONMENT_BEFORE.txt',
               'CONDA_ENVIRONMENT_AFTER.json', 'CONDA_ENVIRONMENT_AFTER_EXPLICIT.txt', 'PIP_ENVIRONMENT_AFTER.txt')


def verify_blocked_before_gpu(directory, read):
    failure = read(directory / 'FAILED.json')
    assert failure['stage'] == '2' and failure['original_input_protection_passed']
    assert failure['automatic_retry'] is False and failure['next_stage_started'] is False
    assert 'Selected GPU admission blocked; no retry/switch/signals' in failure['error']
    assert 'write(base / f\'root_{mode}.GPU_ADMISSION.json\', idle_gpu(gpu[\'uuid\']))' in failure['error']
    for name in ('root_weights.GPU_ADMISSION.json', 'logs/root_weights.launch.json', 'robust_weights.hdf5',
                 'WEIGHTS_COMPLETED.json', 'root_train.GPU_ADMISSION.json', 'logs/root_train.launch.json',
                 'TRAINING_COMPLETED.json', 'AUDIT.json', 'COMPLETED.json'):
        assert not (directory / name).exists(), 'Original stage 2 already has GPU or completion evidence: ' + name
    assert read(directory / 'ENVIRONMENT_READY.json')['passed']
    assert read(directory / 'config.json')['stage'] == '2'
    assert read(directory / 'config.json')['seed'] == 901
    assert read(directory / 'config.json')['main_updates'] == 20000
    return failure


def completed_bc(controller, common):
    verified = {}
    for stage, count in (('1A', 5), ('1B', 30)):
        base = controller.OUTPUTS / controller.STAGE_OUTPUTS[stage]
        receipt = common.read(base / 'COMPLETED.json')
        assert receipt['completed'] and receipt['stage'] == stage and receipt['workers_audited'] == count
        assert common.sha(base / 'RESULT.json') == receipt['result_sha256']
        assert common.sha(base / 'REPORT_ZH.md') == receipt['report_sha256']
        verified[stage] = common.sha(base / 'COMPLETED.json')
    return verified


def validate_root_environment(controller, common):
    ready = common.read(ORIGINAL_OUTPUT / 'ENVIRONMENT_READY.json')
    root_python = controller.ROOT_PREFIX / 'bin/python'
    assert Path(ready['prefix']).resolve() == controller.ROOT_PREFIX.resolve() and root_python.is_file()
    command = [str(root_python), '-B', '-m', 'pip', 'freeze', '--all']
    env = controller.process_env(prefix=controller.ROOT_PREFIX)
    current = subprocess.check_output(command, env=env, encoding='utf-8').splitlines()
    expected = (ORIGINAL_OUTPUT / 'PIP_ENVIRONMENT_AFTER.txt').read_text(encoding='utf-8').splitlines()
    assert current == expected, 'ROOT environment package list differs from the successful installation'
    return ready


def run_workers(controller, common, resources, base, gpu, policy, job_file):
    phases = {}
    for mode in ('prepare', 'weights', 'train', 'audit'):
        limits = resources.limits_snapshot()
        is_gpu = mode in ('weights', 'train')
        if is_gpu:
            # A single ROOT worker, with the same complete idle check as the original run.
            reasons = resources.tier_allowed(1, limits, policy['resource_limits'], 2 * 1024 ** 3,
                                             None, gpu['memory_total_MiB'])
            assert not reasons, reasons
            common.write(base / f'root_{mode}.GPU_ADMISSION.json', common.idle_gpu(gpu['uuid']))
        command = controller.launch_python('root_trial.py', mode, job_file, controller.ROOT_PREFIX)
        env = controller.process_env(gpu['uuid'] if is_gpu else None, controller.ROOT_PREFIX, jax=mode == 'weights')
        print('Stage 2', mode, flush=True)
        phases[mode] = controller.monitor_jobs(base, ['root_' + mode], [command], [env],
                                               limits, policy, gpu, 'root_' + mode)
        if mode == 'prepare':
            for name in PREPARED:
                assert common.read(base / name) == common.read(ORIGINAL_OUTPUT / name), 'Preparation mismatch: ' + name
        elif mode == 'weights':
            assert not phases[mode]['pressure_warnings'], 'Resource pressure during weights, output preserved'
    return phases


def run(controller, common, resources, gpu):
    frozen = common.verify_package()
    policy = common.read(ORIGINAL_CODE / 'config_template.json')
    common.check_environment_protection(common.read(controller.STAGE0 / 'ENVIRONMENT_BEFORE.json'))
    old_env = common.existing_environment_snapshot()
    project = common.verify_project(controller.PROJECT, frozen['project_manifest_sha256'])
    stage0 = controller.stage0_verified()
    bc = completed_bc(controller, common)
    verify_blocked_before_gpu(ORIGINAL_OUTPUT, common.read)
    ready = validate_root_environment(controller, common)
    old_files = {name: common.sha(ORIGINAL_OUTPUT / name) for name in
                 ('FAILED.json', 'STARTED.json', 'root_job.json', 'AUTHORIZATION.json') + PREPARED + ENVIRONMENT}
    old_start = common.read(ORIGINAL_OUTPUT / 'STARTED.json')
    assert old_start['code_freeze_sha256'] == ORIGINAL_FREEZE_SHA256
    old_job = common.read(ORIGINAL_OUTPUT / 'root_job.json')
    assert old_job['task'] == 'walker2d-morph' and old_job['seed'] == 901 and old_job['stage'] == '2'
    assert Path(old_job['project']).resolve() == controller.PROJECT.resolve()
    assert common.read(ORIGINAL_OUTPUT / 'AUTHORIZATION.json')['job_sha256'] == old_files['root_job.json']
    assert gpu['uuid'] == old_job['gpu_uuid'] == old_start['GPU']['uuid'] == stage0['selected_GPU_UUID']

    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / 'logs').mkdir()
    start = time.monotonic()

    def protection():
        controller.check_inputs(frozen, project, old_env)
        for name, digest in old_files.items():
            assert common.sha(ORIGINAL_OUTPUT / name) == digest, name
        assert completed_bc(controller, common) == bc
        validate_root_environment(controller, common)

    try:
        limits = resources.limits_snapshot()
        limits['nproc_all'] = int(subprocess.check_output(['nproc', '--all'], text=True).strip())
        host_before, affinity_before = resources.cpu_ticks(), resources.cpu_ticks(limits['affinity'])
        session = subprocess.check_output(['tmux', 'display-message', '-p', '#S'], text=True).strip()
        launch = f'bash {Path(__file__).resolve().with_name("run_in_tmux.sh")} --registered'
        common.write(OUTPUT / 'STARTED.json', dict(stage='2', time=common.utc(), session=session,
                     selected_GPU=gpu, nvidia_smi=common.full_smi(), limits=limits,
                     user_confirmed_registered=True, site_GPU_process_limit=3, own_GPU_process_limit=1))
        common.write(OUTPUT / 'REUSE.json', dict(original_output=str(ORIGINAL_OUTPUT), original_artifact_sha256=old_files,
                     ROOT_environment=ready, stage0=stage0, BC_completion_sha256=bc,
                     original_training_started=False, reused_installed_environment=True))
        for name in ENVIRONMENT:
            shutil.copyfile(ORIGINAL_OUTPUT / name, OUTPUT / name)
        job = dict(old_job, controller_pid=os.getpid(), output=str(OUTPUT))
        job_file = OUTPUT / 'root_job.json'
        common.write(job_file, job)
        common.write(OUTPUT / 'AUTHORIZATION.json', dict(stage='2', job_sha256=common.sha(job_file),
                     scope='only the unstarted ROOT weights and the original 20k trial', stage3_authorized=False))
        phases = run_workers(controller, common, resources, OUTPUT, gpu, policy, job_file)
        summary = controller.inventory(OUTPUT, start, limits, host_before, affinity_before, session, launch)
        protection()
        controller.root_report(OUTPUT, summary, phases)
        common.write(OUTPUT / 'COMPLETED.json', dict(stage='2', completed=True,
                     status='AWAITING_STAGE3_CONFIRMATION', original_input_protection_passed=True,
                     own_GPU_workers_exited=True, workers_audited=1, stage3_started=False,
                     report_sha256=common.sha(OUTPUT / 'REPORT_ZH.md'), result_sha256=common.sha(OUTPUT / 'RESULT.json'),
                     elapsed_seconds=time.monotonic() - start, ended_utc=common.utc()))
        print('V12 STAGE 2 COMPLETE:', OUTPUT / 'REPORT_ZH.md', flush=True)
        print('AWAITING_STAGE3_CONFIRMATION', flush=True)
        print('Check your GPU processes have ended, then remove your own registration entry.', flush=True)
    except BaseException:
        error = traceback.format_exc()
        try:
            protection()
            protected = True
        except BaseException:
            protected = False
        common.write(OUTPUT / 'FAILED.json', dict(stage='2', error=error, time=common.utc(),
                     input_protection_passed=protected, automatic_retry=False, next_stage_started=False))
        raise


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--registered', action='store_true', required=True,
                        help='confirm you have already registered your own GPU usage entry')
    parser.parse_args()
    assert sys.platform == 'linux' and os.environ.get('TMUX') and os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert sys.version_info[:3] == (3, 8, 13)
    assert Path(sys.prefix).resolve() == (Path.home() / 'conda-envs/offdynamics38').resolve()
    code_hash = hashlib.sha256((ORIGINAL_CODE / 'FROZEN.json').read_bytes()).hexdigest()
    assert code_hash == ORIGINAL_FREEZE_SHA256, 'The original installed Stage 1/2 package differs from release 2'
    sys.path.insert(0, str(ORIGINAL_CODE))
    import common
    import controller
    import resources
    assert Path(controller.__file__).resolve() == (ORIGINAL_CODE / 'controller.py').resolve()
    lock = os.open(str(ORIGINAL_CODE), os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not OUTPUT.exists(), 'Continuation output exists. Preserve it and review its status.'
        common.verify_package()
        # Check the designated GPU before creating the continuation output directory.
        gpu = common.idle_gpu(common.read(ORIGINAL_CODE / 'config_template.json')['resource_admission']['expected_uuid'])
        run(controller, common, resources, gpu)
    finally:
        os.close(lock)


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'Controller received signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupt)
    main()
