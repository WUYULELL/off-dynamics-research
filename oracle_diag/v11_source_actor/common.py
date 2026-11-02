"""V11 Stage0 then the authorized Hopper Stage1; unchanged O1 dependencies."""
import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
O1_CODE = ROOT / 'oracle_diag/v9_hopper_label_headroom/stage2'
sys.path.insert(0, str(O1_CODE))
import runtime as r
sys.path.insert(0, str(HERE))

OUT = ROOT / 'outputs/v11_source_actor_20261101/stage0'
STAGE1 = OUT.parent / 'stage1'
WALKER = ROOT / 'outputs/v9_hopper_label_headroom_20261101/walker_stage3'
TASK = 'hopper-morph'
TRIALS = [('wa0_parity_s201', 'SA-all', 0., 10000),
          ('SA_all_s201', 'SA-all', 1., 20000),
          ('SA_top_s201', 'SA-top', 1., 20000)]

def settings():
    return r.read(OUT / 'config.json')

def guard():
    r.check_frozen()
    c = settings()
    assert c['scope'] == 'V11_Stage0_then_Hopper_Stage1_then_stop'
    assert c['stage1_authorized'] and not c['stage2_authorized']
    assert c['actor_loss'] == 'advantage_weighted_action_MSE'
    for p, h in r.read(OUT / 'FROZEN.json')['sha256'].items():
        assert r.sha(p) == h, 'Frozen input changed: ' + p

def protect(label, create=False):
    path = OUT / 'protected_manifest.json'
    if create:
        original = r.read(WALKER / 'protected_manifest.json')['sha256']
        changed = [p for p, h in original.items() if not Path(r.winpath(p)).exists() or r.sha(p) != h]
        assert not changed, changed[:5]
        paths = list(WALKER.rglob('*'))
        for j in r.read(WALKER / 'training_plan.json')['jobs']:
            paths += list(Path(j['out']).rglob('*'))
        original.update({str(p): r.sha(p) for p in paths if p.is_file()})
        r.write(path, dict(sha256=original, count=len(original), time=time.time(),
            captured_after_Walker_exit=True))
    records = r.read(path)['sha256']
    changed = [p for p, h in records.items() if not Path(r.winpath(p)).exists() or r.sha(p) != h]
    receipt = dict(passed=not changed, count=len(records), changed=changed, time=time.time())
    r.write(OUT / ('protected_' + label + '.json'), receipt)
    assert not changed, changed[:5]
    return receipt

def child(script, args, name, cpu):
    command = [str(r.PYTHON), '-B', str(HERE / script)] + list(args)
    en = os.environ.copy()
    en.update(CUDA_VISIBLE_DEVICES='-1' if cpu else '0', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    en.pop('ODRL_PAPER_STYLE_SOURCE_TARGET', None)
    with open(r.winpath(OUT / 'logs' / (name + '.stdout.log')), 'xb') as so, \
         open(r.winpath(OUT / 'logs' / (name + '.stderr.log')), 'xb') as se:
        p = subprocess.Popen(command, cwd=str(ROOT), env=en, stdout=so, stderr=se,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        r.write(OUT / 'active_child.json', dict(pid=p.pid, command=command, name=name,
            cpu=cpu, time=time.time()), replace=True)
        code = p.wait()
    assert code == 0, (name, code, 'Preserve evidence; no automatic retry or retraining')

def processes():
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"
    value = subprocess.check_output(['powershell', '-NoProfile', '-NonInteractive', '-Command', command],
        creationflags=subprocess.CREATE_NO_WINDOW).decode('utf-8-sig').strip()
    data = json.loads(value) if value else []
    return data if isinstance(data, list) else [data]

def competing_processes():
    own = str(HERE / 'pipeline.py').lower()
    result = []
    for p in processes():
        cmd = (p.get('CommandLine') or '').lower()
        if p['ProcessId'] == os.getpid() or own in cmd:
            continue
        if 'oracle_diag' in cmd and any(s in cmd for s in ('train.py', 'pipeline.py', 'trial.py')):
            result.append(p)
    return result

def tree_digest(value):
    """Deterministic digest of every tensor, optimizer slot and scheduler field."""
    import torch
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous().numpy()
            h.update(repr((str(a.dtype), a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=lambda k: (str(type(k)), str(k))):
                h.update(repr(k).encode()); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for a in x: visit(a)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()

def full_finite(value):
    import math
    import torch
    if torch.is_tensor(value): return bool(torch.isfinite(value).all())
    if isinstance(value, dict): return all(full_finite(v) for v in value.values())
    if isinstance(value, (tuple, list)): return all(full_finite(v) for v in value)
    if isinstance(value, float): return math.isfinite(value)
    return True

def scientific_state(model):
    d = model.checkpoint()
    return {k: v for k, v in d.items() if k != 'config'}

def formal_jobs():
    jobs = r.read(STAGE1/'training_plan.json')['jobs']
    assert len(jobs) == 20
    assert [(j['variant'],j['seed']) for j in jobs] == [(v,s) for v in ('SA-all','SA-top') for s in range(201,211)]
    return jobs

def critic_state(model):
    return {k: v for k, v in scientific_state(model).items()
            if k in ('V', 'Q', 'target_Q', 'V_optimizer', 'Q_optimizer', 'step')}
