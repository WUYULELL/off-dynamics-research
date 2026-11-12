"""Handoff from the original queue only before its first GPU release."""
import os
from pathlib import Path
import signal
import time

from stage3_setup import HOME_DIR, ORIGINAL_OUT, read, sha
from stage3_setup import SHARED50_CODE, SHARED50_FREEZE, SHARED50_OUT, attempt_output


ORIGINAL_CODE = HOME_DIR / 'v12_linux/stage3_20261108'
ORIGINAL_FREEZE = '098f0585f9895f41774d7b5665583cdadbb47f2694a9b8e6dd58b1379cb3025e'


def verify_original_code():
    verify_code(ORIGINAL_CODE, ORIGINAL_FREEZE)


def verify_code(code, expected):
    freeze = code / 'FROZEN.json'
    assert sha(freeze) == expected, 'Previous stage 3 release does not match the expected version'
    for name, digest in read(freeze)['files'].items():
        assert sha(code / name) == digest, name


def require_no_release():
    gates = ORIGINAL_OUT / 'gates'
    assert not (gates.exists() and any(gates.glob('*.json'))), 'Original queue already released GPU work; leave it running'
    assert not (ORIGINAL_OUT / 'weights/robust_weights.hdf5').exists()
    assert not (ORIGINAL_OUT / 'ROOT_weights.COMPLETED.json').exists()
    assert not (ORIGINAL_OUT / 'STAGE3_COMPLETED.json').exists()


def proc(pid):
    base = Path(f'/proc/{pid}')
    try:
        fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        cmd = (base / 'cmdline').read_bytes().split(b'\0')[:-1]
        return dict(start=int(fields[19]), state=fields[0], uid=base.stat().st_uid, cmd=cmd)
    except FileNotFoundError:
        return None


def same_process(pid, original):
    current = proc(pid)
    if current and current['state'] != 'Z' and current['start'] == original['start']:
        assert current['uid'] == os.getuid() and current['cmd'] == original['cmd']
        return current
    return None


def stop_waiting_original():
    if not ORIGINAL_OUT.exists():
        return dict(original_found=False, old_GPU_released=False, signalled=False)
    verify_original_code()
    require_no_release()
    start_file = ORIGINAL_OUT / 'STARTED.json'
    original_start = read(start_file)
    pid = original_start['controller_pid']
    assert isinstance(pid, int) and pid > 1
    previous = proc(pid)
    receipt = dict(original_found=True, old_GPU_released=False, old_PID=pid, start_sha256=sha(start_file), signalled=False)
    if not previous or previous['state'] == 'Z':
        return receipt
    cmd = [b'-u', b'-B', os.fsencode(ORIGINAL_CODE / 'stage3_controller.py'), b'--registered']
    assert previous['uid'] == os.getuid() and previous['cmd'][1:] == cmd, 'Recorded PID is not the original stage 3 controller'
    assert previous['state'] not in ('T', 't'), 'Original controller is already suspended'
    assert (ORIGINAL_OUT / 'weights/ROOT_INPUTS.json').is_file(), 'Original queue has not reached the GPU weights gate'
    assert same_process(pid, previous)
    os.kill(pid, signal.SIGSTOP)
    stopping = False
    try:
        for _ in range(100):
            current = same_process(pid, previous)
            if current is None:
                require_no_release()
                return receipt
            if current['state'] == 'T':
                break
            time.sleep(.05)
        else:
            raise RuntimeError('Original controller did not reach a verified stopped state')
        require_no_release()
        os.kill(pid, signal.SIGTERM)
        stopping = True
    finally:
        if same_process(pid, previous):
            os.kill(pid, signal.SIGCONT)
    assert stopping
    for _ in range(300):
        if same_process(pid, previous) is None:
            require_no_release()
            return dict(receipt, signalled=True, old_proc_start_ticks=previous['start'])
        time.sleep(.1)
    raise RuntimeError('Original controller has not exited; new queue did not start')


def check_preselection50():
    assert not any(attempt_output(n, SHARED50_OUT).exists() for n in range(1, 100)), (
        '50% queue has entered execution; leave it running')
    require_no_release()
    start = ORIGINAL_OUT / 'STARTED.json'
    if start.is_file():
        original = proc(read(start)['controller_pid'])
        if original is not None:
            assert original['state'] not in ('T', 't'), 'Original queue handoff in progress; leave both older processes alone'


def stop_preselection50(pid):
    """Replace only the preceding launcher's CPU-side GPU selection loop."""
    verify_code(SHARED50_CODE, SHARED50_FREEZE)
    check_preselection50()
    old = proc(pid)
    if old is None or old['state'] == 'Z':
        return dict(PID=pid, signalled=False)
    command = old['cmd']
    prefix = [b'-u', b'-B', os.fsencode(SHARED50_CODE / 'stage3_controller.py'), b'--registered']
    assert old['uid'] == os.getuid() and command[1:5] == prefix
    assert len(command) > 5 and all(index in (b'0', b'1', b'2', b'3') for index in command[5:]), (
        'Only a verified 50% initial selector is eligible for replacement')
    assert old['state'] in ('S', 'R') and same_process(pid, old)
    os.kill(pid, signal.SIGSTOP)
    try:
        for _ in range(100):
            current = same_process(pid, old)
            if current is None:
                check_preselection50()
                return dict(PID=pid, signalled=False)
            if current['state'] == 'T':
                break
            time.sleep(.05)
        else:
            raise RuntimeError('50% controller did not reach a verified stopped state')
        check_preselection50()
        os.kill(pid, signal.SIGTERM)
    finally:
        if same_process(pid, old):
            os.kill(pid, signal.SIGCONT)
    for _ in range(300):
        if same_process(pid, old) is None:
            check_preselection50()
            return dict(PID=pid, proc_start_ticks=old['start'], signalled=True)
        time.sleep(.1)
    raise RuntimeError('50% controller has not exited; no new GPU work was released')
