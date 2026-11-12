"""Handoff from the original queue only before its first GPU release."""
import os
from pathlib import Path
import signal
import time

from stage3_setup import HOME_DIR, ORIGINAL_OUT, read, sha


ORIGINAL_CODE = HOME_DIR / 'v12_linux/stage3_20261108'
ORIGINAL_FREEZE = '098f0585f9895f41774d7b5665583cdadbb47f2694a9b8e6dd58b1379cb3025e'


def verify_original_code():
    freeze = ORIGINAL_CODE / 'FROZEN.json'
    assert sha(freeze) == ORIGINAL_FREEZE, 'Original stage 3 release does not match the expected version'
    for name, digest in read(freeze)['files'].items():
        assert sha(ORIGINAL_CODE / name) == digest, name


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
