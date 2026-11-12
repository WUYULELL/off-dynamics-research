"""Verified imports and fixed paths for the V12 Walker stage 3 jobs."""
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
HOME_DIR = Path.home()
LEGACY = HOME_DIR / 'v12_linux/stage1_2_20261107'
LEGACY_FREEZE = 'ff6bdb1e4014bd1fea176a25d5040dca8f82b0149d34e9767427d99a04d4ce2d'
OUT = HOME_DIR / 'v12_linux/outputs/stage3_20261108'
STAGE2 = HOME_DIR / 'v12_linux/outputs/stage2_root_20261108_continue1'
TASK = 'walker2d-morph'
ROOT_COMMIT = '0ae9c05d70ed369c16732c71abda59b54b559da9'
GPU_UUID = 'GPU-965aff83-8d85-646b-eb0e-042db35f2e88'
SEEDS = tuple(range(501, 506))
STEPS = {'ROOT': 1000000, 'T': 500000, 'BC-all': 200000}
EVAL_EVERY = {'ROOT': 10000, 'T': 10000, 'BC-all': 5000}
CAP = 3


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
