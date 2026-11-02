"""Isolated v4 paths; old code and results are read-only dependencies."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
V3_CODE = HERE.parent / 'v3'
OUT = HERE.parent / 'results/v4_20261011'
V3_OUT = HERE.parent / 'results/v3_20261008'
PYTHON = Path('D:/ANACONDA/envs/offdynamics/python.exe')
sys.dont_write_bytecode = True
sys.path.insert(0, str(V3_CODE))
from support_v3 import read, write, sha, fresh, winpath
import hashlib
import json
import time

METHODS = ('ORACLE-ONLY', 'SOURCE-ONLY', 'TARGET-64', 'ORACLE-W1')
SINGLE_SOURCE = ('ORACLE-ONLY', 'SOURCE-ONLY')
ALL_METHODS = ('TARGET',) + METHODS
DRAW_COUNTS = {'TARGET': (0, 128), 'TARGET-64': (0, 64),
               'ORACLE-ONLY': (128, 0), 'SOURCE-ONLY': (128, 0), 'ORACLE-W1': (64, 64)}


def array_hash(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for net in (model.policy, model.v_func, model.q_funcs):
        for value in net.state_dict().values():
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def oracle_receipt():
    gate = read(V3_OUT / 'stage_a_completed.json')
    path = Path(gate['oracle_receipt'])
    assert sha(path) == gate['oracle_receipt_sha256']
    return path


def protected_check():
    manifest = read(OUT / 'protected_manifest.json')
    for index, (path, expected) in enumerate(manifest['sha256'].items()):
        assert sha(path) == expected, 'Protected file changed: ' + path
        if index and index % 1000 == 0:
            print('Protected files checked:', index, flush=True)
    return len(manifest['sha256'])


def check_approval(kind):
    if kind not in ('smoke', 'formal'):
        raise ValueError(kind)
    auth_path = OUT / ('gpu_' + kind + '_authorization.json')
    if not auth_path.exists():
        raise RuntimeError('Explicit v4 ' + kind + ' GPU authorization is required; v3 authorization is not reused.')
    auth = read(auth_path)
    assert auth['authorized'] is True and auth['scope'] == kind
    assert auth['protocol_sha256'] == sha(OUT / 'protocol.json')
    ready = read(OUT / 'cpu_ready.json')
    assert ready['passed']
    for path, expected in ready['code_sha256'].items():
        assert sha(path) == expected, 'Code changed after CPU review: ' + path
    if kind == 'formal':
        assert read(OUT / 'smoke_audited.json')['passed']
        assert read(OUT / 'target_reuse_verified.json')['passed']
    return ready
