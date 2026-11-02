# -*- coding: utf-8 -*-
"""Prepare one new recovery attempt. Never modify previous code or results."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD_CODE = ROOT / 'oracle_diag/coverage_oracle_r13'
PRIOR = ROOT / 'oracle_diag/results/coverage_oracle_20261021_r13'
OUT = ROOT / 'oracle_diag/results/coverage_oracle_20261023_r13_resume2'
DOC = ROOT / 'outputs/research_idea_protocol_20261023_r13_resume2'
PAUSE = ROOT / 'outputs/research_idea_protocol_20261021_r13/pause_20261022_183640'

def wp(p):
    s = str(Path(p).resolve())
    return s if os.name != 'nt' or s.startswith('\\\\?\\') else '\\\\?\\' + s

def read(p):
    with open(wp(p), encoding='utf-8-sig') as f:
        return json.load(f)

def sha(p):
    h = hashlib.sha256()
    with open(wp(p), 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(wp(p), 'x', encoding='utf-8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)

def main():
    assert not OUT.exists() and not DOC.exists()
    for n in ('learner.py', 'audit.py', 'train.py', 'decision.py'):
        assert not (HERE / n).exists()
        shutil.copyfile(wp(OLD_CODE / n), wp(HERE / n))
    text = (OLD_CODE / 'runtime.py').read_text(encoding='utf-8')
    text = text.replace("OUT = ROOT / 'oracle_diag/results/coverage_oracle_20261021_r13'",
                        "OUT = ROOT / 'oracle_diag/results/coverage_oracle_20261023_r13_resume2'")
    text = text.replace("PRIOR = ROOT / 'oracle_diag/results/coverage_oracle_20261018_r12'",
                        "PRIOR = ROOT / 'oracle_diag/results/coverage_oracle_20261021_r13'")
    text = text.replace("DOC = ROOT / 'outputs/research_idea_protocol_20261021_r13'",
                        "DOC = ROOT / 'outputs/research_idea_protocol_20261023_r13_resume2'")
    begin = text.index('def check_frozen():')
    end = text.index('def configure_cpu():')
    guard = '''def check_frozen():
    a = read(OUT / 'stage_authorization.json')
    assert a['scope'] == 'R13_resume_G205_100k_then_R201_205_only'
    assert a['protocol_sha256'] == sha(OUT / 'protocol.json') == sha(PRIOR / 'protocol.json')
    assert a['explicit_user_resume_20261023'] and a['new_GPU_jobs'] == 6
    assert all(a[k] is False for k in ('C_authorized', 'D_authorized', 'task2_authorized',
        'original_formal_authorized', 'estimator_development_authorized', 'extra_T_search_authorized', 'T128_authorized'))
    for manifest in ('code_frozen.json', 'dependency_hashes.json'):
        for p, h in read(OUT / manifest)['sha256'].items():
            assert sha(p) == h, 'Frozen recovery file changed: ' + p
    for manifest in ('code_frozen.json', 'dependency_hashes.json'):
        for p, h in read(PRIOR / manifest)['sha256'].items():
            assert sha(p) == h, 'Original frozen file changed: ' + p

def validate_spec(job):
    assert job['phase'] == 'confirmation' and job['task'] == 'walker2d-morph'
    assert job['temp'] == 10 and job['steps'] == 500000 and job['ratio'] == .05
    assert (job['arm'] == 'G' and job['seed'] == 205) or (job['arm'] == 'R' and job['seed'] in (201, 202, 203, 204, 205))
    assert OUT in Path(job['out']).resolve().parents

def check_job(job, for_launch=True):
    check_frozen(); validate_spec(job)
    assert job in read(OUT / 'training_plan.json')['jobs']
    assert read(OUT / 'cpu_acceptance.json')['passed']
    if for_launch:
        assert not (OUT / 'STAGE_COMPLETED.json').exists()

'''
    text = text[:begin] + guard + text[end:]
    with open(HERE / 'runtime.py', 'x', encoding='utf-8', newline='\n') as f:
        f.write(text)
    OUT.mkdir(); DOC.mkdir()
    for n in ('jobs', 'logs', 'scores', 'random_masks'):
        (OUT / n).mkdir()
    for n in ('protocol.json', 'PRE_RESULT_COMMITMENT.json', 'R_mask_receipts.json'):
        shutil.copyfile(wp(PRIOR / n), wp(OUT / n))
    for folder in ('scores', 'random_masks'):
        for f in (PRIOR / folder).iterdir():
            if f.is_file():
                shutil.copyfile(wp(f), wp(OUT / folder / f.name))
    proof = read(PAUSE / 'PAUSE_VERIFICATION.json')
    assert proof['overall_saved_state_verified'] and proof['checkpoint_step'] == 100000
    assert sha(proof['checkpoint']) == proof['checkpoint_sha256']
    old_plan = read(PRIOR / 'training_plan.json')['jobs']
    reused, jobs = [], []
    for spec in old_plan:
        if spec['id'] in proof['completed_audited_jobs']:
            audit = Path(spec['out']) / 'audit.json'
            a = read(audit); assert a['passed']
            reused.append(dict(job=spec, audit=str(audit), audit_sha256=sha(audit),
                completed_sha256=sha(Path(spec['out']) / 'completed.json'),
                final_model_sha256=sha(read(Path(spec['out']) / 'completed.json')['final_model'])))
        else:
            j = dict(spec); j['out'] = str(OUT / 'runs' / j['id'])
            jobs.append(j); write(OUT / 'jobs' / (j['id'] + '.json'), j)
    assert len(reused) == 14 and len(jobs) == 6
    assert [(j['arm'], j['seed']) for j in jobs] == [('G', 205)] + [('R', s) for s in range(201, 206)]
    write(OUT / 'training_plan.json', dict(jobs=jobs, new_GPU_jobs=6, original_logical_jobs=20,
        new_updates=2900000, resumed_G_new_updates=400000, fresh_R_updates=2500000,
        original_plan=str(PRIOR / 'training_plan.json'), original_plan_sha256=sha(PRIOR / 'training_plan.json')))
    write(OUT / 'reuse_receipts.json', dict(runs=reused, complete_audited=14, no_retraining=True))
    write(OUT / 'recovery.json', dict(checkpoint=proof['checkpoint'], checkpoint_sha256=proof['checkpoint_sha256'],
        checkpoint_step=100000, original_run=str(Path(proof['checkpoint']).parent),
        pause_proof=str(PAUSE / 'PAUSE_VERIFICATION.json'), pause_proof_sha256=sha(PAUSE / 'PAUSE_VERIFICATION.json'),
        prefix_through=100000, old_overlap_end=109000, exact_float32_overlap_required=True,
        expected_streams=proof['sampling_stream_reconstruction_verified'], expected_GPU='NVIDIA GeForce RTX 2060 SUPER',
        expected_GPU_uuid='GPU-4893eae9-0a98-dde8-c0ef-bad66a87d788', scientific_protocol_changed=False,
        original_learner=dict(path=str(OLD_CODE / 'learner.py'), sha256=sha(OLD_CODE / 'learner.py'))))
    write(OUT / 'stage_authorization.json', dict(scope='R13_resume_G205_100k_then_R201_205_only',
        protocol_sha256=sha(OUT / 'protocol.json'), explicit_user_resume_20261023=True,
        user_instruction='我没有更换显卡，继续实验', new_GPU_jobs=6, logical_total=20,
        C_authorized=False, D_authorized=False, task2_authorized=False, original_formal_authorized=False,
        estimator_development_authorized=False, extra_T_search_authorized=False, T128_authorized=False,
        no_extra_training_after_stage=True, committed=time.time()))
    protected = dict(read(PRIOR / 'protected_manifest.json')['sha256'])
    for root in (PRIOR, OLD_CODE, ROOT / 'outputs/research_idea_protocol_20261021_r13', ROOT / 'oracle_diag/coverage_oracle_r13_resume1', ROOT / 'oracle_diag/results/coverage_oracle_20261023_r13_resume1', ROOT / 'outputs/research_idea_protocol_20261023_r13_resume1'):
        for f in root.rglob('*'):
            if f.is_file() and '__pycache__' not in str(f):
                protected[str(f)] = sha(f)
    bad = [p for p, h in protected.items() if not os.path.exists(wp(p)) or sha(p) != h]
    assert not bad, bad
    write(OUT / 'protected_manifest.json', dict(count=len(protected), sha256=protected, includes_original_4006=True))
    delta = {n: dict(before_sha256=sha(OLD_CODE / n), after_sha256=sha(HERE / n)) for n in
             ('learner.py', 'audit.py', 'train.py', 'decision.py', 'runtime.py')}
    assert all(delta[n]['before_sha256'] == delta[n]['after_sha256'] for n in ('learner.py', 'audit.py', 'train.py', 'decision.py'))
    write(OUT / 'engineering_repair.json', dict(delta=delta,
        additions='mixed-source full-state recovery, two samplers, prefix/overlap provenance, authorized six-job queue',
        unchanged='learner, fresh-R train, audit, decisions, evaluation namespace, all scientific settings',
        old_files_protected=len(protected), no_old_writes=True))
    write(OUT / 'active_execution.json', dict(code=str(HERE), results=str(OUT), queue_script=str(HERE / 'pipeline.py'),
        queue_status=str(OUT / 'queue_status.json'), training_plan=str(OUT / 'training_plan.json'),
        original=str(PRIOR), original_queue_must_not_restart=True, resume_step=100000))
    for f in HERE.glob('*.py'):
        ast.parse(f.read_text(encoding='utf-8'))
    write(OUT / 'code_frozen.json', dict(sha256={str(f):sha(f) for f in sorted(HERE.glob('*.py'))},
        preparation_only=True, GPU_not_started=True))
    files = [OUT / n for n in ('protocol.json', 'stage_authorization.json', 'training_plan.json',
        'recovery.json', 'reuse_receipts.json', 'R_mask_receipts.json', 'PRE_RESULT_COMMITMENT.json')]
    files += list((OUT / 'jobs').iterdir()) + list((OUT / 'scores').iterdir()) + list((OUT / 'random_masks').iterdir())
    write(OUT / 'dependency_hashes.json', dict(sha256={str(f):sha(f) for f in files}))
    print('Recovery preparation complete; 14 reused, 6 remaining; protected=%d' % len(protected), flush=True)

if __name__ == '__main__':
    main()
