"""Process queue and shared controller steps (adaptive-source batch).

* only the GPUs the relay put in CUDA_VISIBLE_DEVICES are used; every worker sees exactly one of them;
* at most max_per_gpu workers per GPU; no new worker is started while system memory use is >= mem_limit
  (0.75), so concurrency drops until memory is available again (events are logged);
* a worker killed by an outside signal (negative return code, e.g. OOM kill) is rerun from step 0 with the same
  seed in a new folder, at most external_retry_max (3) times; a code error / NaN (positive return code or no
  COMPLETED.json) stops new launches, running workers finish, and the phase reports the failure;
* rerun decisions never depend on returns.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from asrc_common import CODE, HOME, mem_used_fraction, read_json, rel, scrub, utc, write_json

OUT = CODE/'outputs'


def log(cfg, *a):
    print('[%s %s]' % (cfg['job'], utc()), *[scrub(x) for x in a], flush=True)


def visible_gpus(local):
    if local:
        return ['cpu']
    value = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    gpus = [g.strip() for g in value.split(',') if g.strip() and g.strip() != '-1']
    assert gpus, 'CUDA_VISIBLE_DEVICES is empty or -1: this job needs the GPU(s) the relay assigns'
    return gpus


def setup_mujoco(local):
    """Private copy of the installed mujoco_py (as j003-j012); returns the vendor directory or None (local)."""
    if local:
        return None
    mujoco = HOME/'.mujoco210/mujoco210'
    assert (mujoco/'bin').is_dir(), 'missing ~/.mujoco210/mujoco210'
    from importlib import metadata
    original = Path(metadata.distribution('mujoco-py').locate_file('mujoco_py')).resolve()
    vendor = CODE/'private_runtime'
    if not (vendor/'mujoco_py').is_dir():
        vendor.mkdir(exist_ok=True)
        shutil.copytree(str(original), str(vendor/'mujoco_py'))
    sys.path.insert(0, str(vendor))
    import mujoco_py  # noqa: F401  (builds / loads once here, before workers start)
    assert vendor.resolve() in Path(mujoco_py.__file__).resolve().parents, 'mujoco_py not imported from private copy'
    return str(vendor)


def transfer_learner():
    """Transfer project (read-only): learner directory with offline_offline/iql_unc.py, hashes checked against
    its MANIFEST.json (as j011/j012)."""
    project = HOME/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
    algo = project/'source_snapshot/algo'
    from asrc_common import sha256_file
    files = {str((project/e['path']).resolve()): e for e in read_json(project/'MANIFEST.json')['files']}
    for p in [algo/'offline_offline/iql_dyn.py', algo/'offline_offline/iql_unc.py']:
        e = files[str(p.resolve())]
        assert sha256_file(p) == e['sha256'], 'transfer learner hash mismatch: %s' % p.name
    return str(algo)


def run_queue(cfg, jobs, base, gpus, phase, local=False):
    """jobs: dicts with 'id' and 'worker' (script file name in code/). Returns (done, failed, events)."""
    pending = [dict(j) for j in jobs]
    running, done, failed, events = {}, {}, {}, []
    last_beat, mem_blocked = 0., False
    max_per_gpu = int(cfg.get('max_per_gpu', 6))
    mem_limit = float(cfg.get('mem_limit', .75))
    retry_max = int(cfg.get('external_retry_max', 3))
    gap = 0. if local else float(cfg.get('launch_gap_seconds', 20))
    while pending or running:
        for name, r in list(running.items()):
            code = r['process'].poll()
            if code is None:
                continue
            r['log'].close()
            del running[name]
            ok = code == 0 and (r['folder']/'COMPLETED.json').is_file()
            events.append(dict(time=utc(), event='exit', job=name, gpu=r['gpu'], attempt=r['attempt'],
                               returncode=code, completed=ok))
            if ok:
                done[name] = r['folder']
            elif code < 0 and r['attempt'] < retry_max:
                log(cfg, 'external interruption, rerun from step 0 with the same seed:', name, 'code', code)
                pending.insert(0, dict(r['job'], _attempt=r['attempt']+1))
            else:
                failed[name] = dict(folder=rel(r['folder']), returncode=code, attempt=r['attempt'])
                try:
                    tail = (r['folder']/'stdout.log').read_text(errors='replace').splitlines()[-40:]
                except Exception:
                    tail = []
                log(cfg, 'RUN FAILED', name, 'code', code, '\n'+'\n'.join(tail))
        if failed:
            pending = []
        while pending:
            load = {g: sum(1 for r in running.values() if r['gpu'] == g) for g in gpus}
            gpu = min(gpus, key=lambda g: load[g])
            if load[gpu] >= max_per_gpu:
                break
            used = mem_used_fraction()
            if used is not None and used >= mem_limit and running:
                if not mem_blocked:
                    events.append(dict(time=utc(), event='memory_backoff', used=used, running=len(running)))
                    log(cfg, 'memory use %.0f%% >= %.0f%%: not starting new workers' % (100*used, 100*mem_limit))
                mem_blocked = True
                break
            if mem_blocked:
                events.append(dict(time=utc(), event='memory_ok', used=used, running=len(running)))
                mem_blocked = False
            job = pending.pop(0)
            attempt = job.pop('_attempt', 0)
            folder = base/(job['id'] if attempt == 0 else '%s_retry%d' % (job['id'], attempt))
            folder.mkdir(parents=True)
            job = dict(job, gpu=gpu, output=str(folder), attempt=attempt, code=str(CODE))
            write_json(folder/'job.json', job)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            out = (folder/'stdout.log').open('w')
            process = subprocess.Popen([sys.executable, '-u', '-B', str(CODE/job['worker']), str(folder/'job.json')],
                                       stdout=out, stderr=subprocess.STDOUT, env=env, cwd=str(folder))
            running[job['id']] = dict(job=job, process=process, gpu=gpu, folder=folder, log=out, attempt=attempt)
            events.append(dict(time=utc(), event='start', job=job['id'], gpu=gpu, attempt=attempt))
            if gap:
                time.sleep(gap)
        if time.time()-last_beat > 60:
            last_beat = time.time()
            progress = {}
            for name, r in running.items():
                try:
                    p = read_json(r['folder']/'progress.json')
                    progress[name] = '%s/%s' % (p.get('step'), p.get('total'))
                except Exception:
                    progress[name] = 'starting'
            write_json(OUT/'progress.json', dict(time=utc(), phase=phase, pending=len(pending), running=len(running),
                                                 done=len(done), failed=len(failed), memory_used=mem_used_fraction(),
                                                 progress=progress))
            log(cfg, phase, 'pending', len(pending), 'running', len(running), 'done', len(done), 'failed', len(failed))
        time.sleep(.2 if local else 10)
    write_json(OUT/('events_%s.json' % phase), events)
    return done, failed, events


def collect_evals(folder, steps, every, seed, seed_origin):
    """Last5 etc. from the eval files of a recorded run (BC worker and OTDF worker share the format)."""
    import math
    points = list(range(every, steps+1, every))
    evals = [read_json(Path(folder)/('eval_%07d.json' % p)) for p in points]
    origin = 500000+10*(seed-seed_origin)
    for p, e in zip(points, evals):
        assert e['step'] == p and len(e['scores']) == 10 and e['env_seeds'] == list(range(origin, origin+10))
        assert all(math.isfinite(s) for s in e['scores']) and abs(sum(e['scores'])/10-e['mean']) < 1e-9
    last = evals[-5:]
    lengths = [l for e in last for l in e['episode_lengths']]
    return dict(last5=sum(e['mean'] for e in last)/len(last), final=evals[-1]['mean'],
                best_point=max(e['mean'] for e in evals), length_mean=sum(lengths)/len(lengths),
                length_min=min(lengths), fraction_under_200=sum(l < 200 for l in lengths)/len(lengths),
                last5_points=[e['step'] for e in last], last5_raw_mean=sum(sum(e['raw_returns'])/10 for e in last)/len(last),
                curve=[round(e['mean'], 4) for e in evals])
