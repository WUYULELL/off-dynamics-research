"""Read-only inventory of an interrupted V12 stage 3 queue."""
import os
from pathlib import Path

from stage3_setup import GPU_UUIDS, HERE, HOME_DIR, ORIGINAL_OUT, SEEDS, SHARED25_OUT, STEPS, TASK
from stage3_setup import attempt_output, points, read, sha, verify_new


OLDER = {
    ORIGINAL_OUT: ('stage3_20261108', '098f0585f9895f41774d7b5665583cdadbb47f2694a9b8e6dd58b1379cb3025e'),
    SHARED25_OUT: ('stage3_shared_20261108', '7eb87e1cbe885353e77952c95ea212b04c8baa05332e7020bc62caa79389f3af'),
}


def history():
    candidates = list(OLDER) + [attempt_output(n) for n in range(1, 100)]
    return [path for path in candidates if path.exists()]


def next_attempt():
    previous = [n for n in range(1, 100) if attempt_output(n).exists()]
    number = max(previous, default=0) + 1
    return number, attempt_output(number)


def assert_no_live(allowed=()):
    package_dirs = [HOME_DIR / 'v12_linux' / code for code, _ in OLDER.values()] + [HERE]
    names = {os.fsencode(d / file) for d in package_dirs for file in ('stage3_controller.py', 'stage3_worker.py')}
    active = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        pid = int(process.name)
        if pid == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            args = (process / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        scripts = names.intersection(args)
        if scripts and not scripts.issubset(allowed):
            active.append(pid)
    assert not active, f'Existing stage 3 processes are still alive: {sorted(active)}. Leave them running.'


def verify_history_code(paths):
    verify_new()
    current = read(HERE / 'PROTOCOL.json')
    resource_keys = {'GPU', 'GPU_occupancy_policy', 'recovery'}
    scientific = {k: v for k, v in current.items() if k not in resource_keys}
    for path in paths:
        if path in OLDER:
            name, digest = OLDER[path]
            package = HOME_DIR / 'v12_linux' / name
            assert sha(package / 'FROZEN.json') == digest, package
            for file, value in read(package / 'FROZEN.json')['files'].items():
                assert sha(package / file) == value, file
            expected = package / 'PROTOCOL.json'
        else:
            expected = HERE / 'PROTOCOL.json'
        recorded = path / 'PROTOCOL.json'
        if recorded.is_file():
            assert sha(recorded) == sha(expected), recorded
            assert {k: v for k, v in read(recorded).items() if k not in resource_keys} == scientific


def released_gpu(paths):
    cards = set()
    for path in paths:
        gates = sorted((path / 'gates').glob('*.json'))
        if gates:
            started = read(path / 'STARTED.json')
            for gate in gates:
                grant = read(gate)
                assert grant['controller_pid'] == started['controller_pid']
                assert grant['GPU_UUID'] == started['GPU']['uuid'] and gate.stem in grant['jobs']
            cards.add(started['GPU']['uuid'])
    assert len(cards) <= 1, 'Previous GPU work used multiple devices; stop for review'
    if cards:
        uuid, = cards
        assert uuid in GPU_UUIDS.values()
        return uuid
    return None


def verify_audit(job, audit_path):
    directory = Path(job['output'])
    completed = read(directory / 'COMPLETED.json')
    assert completed['job'] == job and completed['main_steps'] == STEPS[job['group']]
    ck_sha = sha(directory / 'final_checkpoint.pt')
    assert ck_sha == completed['checkpoint_sha256']
    audit = read(audit_path)
    assert audit['passed'] and audit['CPU_only'] and not audit['CUDA_initialized']
    assert (audit['group'], audit['seed'], audit['steps']) == (job['group'], job['seed'], STEPS[job['group']])
    assert audit['completed_sha256'] == sha(directory / 'COMPLETED.json') and audit['checkpoint_sha256'] == ck_sha
    hashes = audit['evaluation_sha256']
    assert set(hashes) == {str(x) for x in points(job['group'])}
    for step, digest in hashes.items():
        assert sha(directory / f'eval_{int(step):07d}.json') == digest


def collect(paths, project, uuid):
    completed, interrupted = {}, []
    for previous in paths:
        assert not (previous / 'STAGE3_COMPLETED.json').exists(), f'Stage 3 already has a completion receipt in {previous}'
        for group, steps in STEPS.items():
            for seed in SEEDS:
                name = f'{group}_s{seed}'
                directory = previous / 'runs' / name
                end = directory / 'COMPLETED.json'
                if not end.is_file():
                    if (previous / 'gates' / (name + '.json')).is_file():
                        interrupted.append(dict(id=name, previous_output=str(directory)))
                    continue
                assert name not in completed, f'More than one completed run for {name}'
                path = previous / 'jobs' / (name + '.json')
                auth = read(previous / 'AUTHORIZATION.json')
                assert auth['scope'] == 'V12_STAGE3_3_GPU_PROCESSES_ONLY' and auth['jobs'][name] == sha(path)
                job = read(path)
                assert (job['id'], job['group'], job['seed'], job['steps'], job['task'], job['stage']) == (name, group, seed, steps, TASK, '3')
                assert Path(job['project']).resolve() == project.resolve() and Path(job['output']).resolve() == directory.resolve()
                assert uuid is not None and job['gpu_uuid'] == uuid
                assert read(previous / 'gates' / (name + '.json'))['GPU_UUID'] == uuid
                done = read(end)
                assert done['job'] == job and done['main_steps'] == steps
                assert done['checkpoint_sha256'] == sha(directory / 'final_checkpoint.pt')
                audit = directory / 'AUDIT.json'
                if not audit.exists():
                    receipts = [p / 'reuse_audits' / (name + '.json') for p in paths]
                    audit = next((p for p in receipts if p.exists()), audit)
                if audit.exists():
                    verify_audit(job, audit)
                completed[name] = dict(job=job, audit=str(audit) if audit.exists() else None)
    return completed, interrupted


def completed_weights(paths):
    candidates = []
    for previous in paths:
        receipt = previous / 'weights/WEIGHTS_COMPLETED.json'
        if receipt.is_file():
            data = read(receipt)
            path = previous / 'weights/robust_weights.hdf5'
            assert data['completed'] and data['sha256'] == sha(path)
            assert data['CUDA_VISIBLE_DEVICES'] == read(previous / 'gates/weights.json')['GPU_UUID']
            inputs = read(previous / 'ROOT_INPUTS.json')
            assert inputs['ROOT_source_next'] == 'upstream_adjacent'
            candidates.append(dict(path=str(path), receipt=data, inputs=inputs))
    assert len({x['receipt']['sha256'] for x in candidates}) <= 1, 'Multiple different completed OT weights; stop for review'
    return candidates[0] if candidates else None
