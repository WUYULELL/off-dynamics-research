"""Verified imports and fixed paths for the V12 Walker stage 3 jobs."""
import hashlib
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
HOME_DIR = Path.home()
LEGACY = HOME_DIR / 'v12_linux/stage1_2_20261107'
LEGACY_FREEZE = 'ff6bdb1e4014bd1fea176a25d5040dca8f82b0149d34e9767427d99a04d4ce2d'
ORIGINAL_OUT = HOME_DIR / 'v12_linux/outputs/stage3_20261108'
SHARED25_OUT = HOME_DIR / 'v12_linux/outputs/stage3_shared_20261108'
SHARED50_OUT = HOME_DIR / 'v12_linux/outputs/stage3_shared50_20261108'
SHARED50_CODE = HOME_DIR / 'v12_linux/stage3_shared50_20261108'
SHARED50_FREEZE = '3c052a3b615be9a7bbc7af582ae5e4c7daabd0321d9b7b78090f21641cb3ee33'
SHARED60_OUT = HOME_DIR / 'v12_linux/outputs/stage3_shared60_20261108'
SHARED60_CODE = HOME_DIR / 'v12_linux/stage3_shared60_20261108'
SHARED60_FREEZE = '6c916a400ba54b860a0a1c5a72abd9b52394c7b4a8c120cdeac178726b518079'
BASE_OUT = HOME_DIR / 'v12_linux/outputs/stage3_switch_20261110'
STAGE2 = HOME_DIR / 'v12_linux/outputs/stage2_root_20261108_continue1'
TASK = 'walker2d-morph'
ROOT_COMMIT = '0ae9c05d70ed369c16732c71abda59b54b559da9'
GPU_UUIDS = {
    0: 'GPU-965aff83-8d85-646b-eb0e-042db35f2e88',
    1: 'GPU-06b314a5-6d68-6c46-4008-802f8a7e9d54',
    2: 'GPU-46bd602a-acec-adb7-9684-4292d55ba65c',
    3: 'GPU-ccef28f6-93fc-9d7d-037e-e8217619672a',
}
SEEDS = tuple(range(501, 506))
STEPS = {'ROOT': 1000000, 'T': 500000, 'BC-all': 200000}
EVAL_EVERY = {'ROOT': 10000, 'T': 10000, 'BC-all': 5000}
CAP = 3


def attempt_output(number, base=BASE_OUT):
    assert 1 <= number <= 99
    return base if number == 1 else base.with_name(f'{base.name}_attempt{number:02d}')


OUT = attempt_output(int(os.environ.get('V12_STAGE3_ATTEMPT', '1')))


def sha(path):
    with Path(path).open('rb') as handle:
        result = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 ** 2), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_new():
    frozen = read(HERE / 'FROZEN.json')
    assert frozen['GPU_process_limit'] == CAP and frozen['upstream_commit'] == ROOT_COMMIT
    for name, digest in frozen['files'].items():
        assert sha(HERE / name) == digest, name
    return frozen


def verify_legacy(directory):
    directory = Path(directory).resolve()
    assert sha(directory / 'FROZEN.json') == LEGACY_FREEZE
    frozen = read(directory / 'FROZEN.json')
    for name, digest in frozen['files'].items():
        assert sha(directory / name) == digest, name
    return frozen


def load_legacy(directory=LEGACY):
    directory = Path(directory).resolve()
    verify_legacy(directory)
    sys.path.insert(0, str(directory))
    import bc
    import common
    import controller
    import resources
    import root_trial
    import science
    for module in (bc, common, controller, resources, root_trial, science):
        assert Path(module.__file__).resolve().parent == directory, module.__file__
    return bc, common, controller, resources, root_trial, science


def points(group):
    return list(range(EVAL_EVERY[group], STEPS[group] + 1, EVAL_EVERY[group]))


def source_mode():
    mode = read(HERE / 'PROTOCOL.json')['ROOT_source_next']
    assert mode == 'upstream_adjacent'
    return mode
