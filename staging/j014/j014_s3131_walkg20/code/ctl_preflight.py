"""j013_adapt_preflight controller.

1  hashes: FROZEN.json, official OTDF files (LF bytes of commit 41e2e62 and their CRLF lab-computer checkout),
   minimal d4rl 1.1, XMLs, D4RL source files (research/data_manifests/d4rl_source_20261121.json), ODRL targets,
   official hopper-morph target; official training arrays rebuilt on the server, [s, a, s'] SHA-256 and rows.
   OT costs are computed here (server side, no lab-computer cost files): official solve_ot via ot_cost.py, run by an
   OT-capable Python (jax + ott-jax + jaxopt + h5py + numpy); if none is found the step is reported as blocked.
   After OT the arrays are rebuilt once more and must match the OT manifest; cost length = source rows.
2  support statistics per task (tau, tau_state, near-layer fractions of all source / of D_ret for rho 0.2/0.8/1.0).
3  unit tests a-d and the per-row layer-weight test (preflight_tests.py).
4  smoke: 7 tasks x {rho0, rho80, rho20T, rho100T, rho80T, rho80R, BC} (+ IQLCAL on the two hopper kinematic
   tasks), 2000 steps + 1 evaluation, seed 2999; finite values and a saved, loadable model; returns never written
   (OTDF worker) or never read (BC worker).
5  throughput: 6 OTDF processes on one GPU, 20000 steps with 2 evaluations (walker and hopper task), estimate of
   T (1M steps) and of one j014 job, split recommendation.
6  normalization table: a fixed grid of raw returns converted with the ODRL references and with OTDF info.py.
   + simulator state-restoration check (j016 prerequisite).
Outputs: PREFLIGHT.json, REPORT_ZH.md, COST_MANIFEST.json, SUPPORT.json, PREPARED_MANIFEST.json.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import asrc_common as C
import asrc_ctl as X
import asrc_data
import asrc_queue as Q

OTDF_GROUPS = ('rho0', 'rho80', 'rho20T', 'rho100T', 'rho80T', 'rho80R')
TIERED = ('rho20T', 'rho100T', 'rho80T', 'rho80R')
PROPORTION = {'rho0': 0., 'rho80': .8, 'rho20T': .2, 'rho100T': 1., 'rho80T': .8, 'rho80R': .8, 'IQLCAL': 0.}


def environment_info():
    import platform
    info = dict(python=platform.python_version(), cpu_count=os.cpu_count(),
                cpu_affinity=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                memory_used_fraction=C.mem_used_fraction())
    for name in ('numpy', 'torch', 'gym', 'h5py', 'yaml', 'tqdm', 'scipy'):
        try:
            m = __import__(name)
            info[name] = getattr(m, '__version__', '?')
        except Exception as e:
            info[name] = 'IMPORT FAILED: %s' % type(e).__name__
    return info


# ----------------------------------------------------------------------------- OT

def find_ot_python(cfg):
    """First Python (job env, then sibling conda envs named in config) that imports jax, ott, jaxopt, h5py, numpy."""
    lc = X.local_cfg()
    cands = []
    if lc and lc.get('ot_python'):
        cands.append(Path(lc['ot_python']))
    cands.append(Path(sys.executable))
    prefix = os.environ.get('CONDA_PREFIX')
    if prefix:
        for name in cfg['ot']['conda_env_candidates']:
            cands.append(Path(prefix).parent/name/'bin'/'python')
    probe = ('import json, importlib\nout = {}\nfor n in ("jax", "jaxlib", "ott", "jaxopt", "h5py", "numpy"):\n'
             '    m = importlib.import_module(n); out[n] = getattr(m, "__version__", "?")\nprint(json.dumps(out))')
    tried = []
    for py in cands:
        if not py.is_file():
            tried.append(dict(python=C.rel(py), result='missing'))
            continue
        env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
        try:
            p = subprocess.run([str(py), '-c', probe], capture_output=True, text=True, timeout=600, env=env)
        except Exception as e:
            tried.append(dict(python=C.rel(py), result='probe error %s' % type(e).__name__))
            continue
        if p.returncode == 0:
            try:
                versions = json.loads(p.stdout.strip().splitlines()[-1])
                tried.append(dict(python=C.rel(py), result='ok', versions=versions))
                return py, versions, tried
            except Exception:
                pass
        tried.append(dict(python=C.rel(py), result='import failed', stderr_tail=C.scrub(p.stderr[-400:])))
    return None, None, tried


class OTPool(threading.Thread):
    """Runs ot_cost.py for every task with at most `parallel` processes (CPU, nice 10)."""

    def __init__(self, cfg, python, requests):
        super().__init__(daemon=True)
        self.cfg, self.python, self.requests = cfg, python, list(requests)
        self.results, self.failed, self.procs = {}, {}, {}
        ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        self.parallel = max(1, min(len(self.requests), ncpu//int(cfg['ot']['cores_per_process'])))
        self.deadline = time.time()+3600*float(cfg['ot']['max_hours'])
        self.timed_out = False

    def run(self):
        pending = list(self.requests)
        while pending or self.procs:
            for task, (p, req, logf) in list(self.procs.items()):
                code = p.poll()
                if code is None:
                    continue
                logf.close()
                del self.procs[task]
                if code == 0 and Path(req['out_manifest']).is_file():
                    self.results[task] = C.read_json(req['out_manifest'])
                else:
                    tail = Path(req['log']).read_text(errors='replace').splitlines()[-30:]
                    self.failed[task] = dict(returncode=code, tail=C.scrub('\n'.join(tail)))
            if time.time() > self.deadline:
                self.timed_out = True
                for task, (p, req, logf) in list(self.procs.items()):
                    p.kill()
                    self.failed[task] = dict(returncode=None, tail='killed: OT time limit')
                self.procs.clear()
                for req in pending:
                    self.failed[req['task']] = dict(returncode=None, tail='not started: OT time limit')
                return
            while pending and len(self.procs) < self.parallel:
                req = pending.pop(0)
                rq = Path(req['request'])
                rq.write_text(json.dumps({k: v for k, v in req.items() if k not in ('request', 'log')}, indent=2), encoding='utf-8')
                env = {k: v for k, v in os.environ.items() if k not in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
                env.update(JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false', CUDA_VISIBLE_DEVICES='')
                logf = open(req['log'], 'w')
                p = subprocess.Popen(['nice', '-n', '10', str(self.python), '-u', '-I', str(C.CODE/'ot_cost.py'), str(rq)],
                                     stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(C.CODE))
                self.procs[req['task']] = (p, req, logf)
            time.sleep(5)


# ----------------------------------------------------------------------------- jobs

def otdf_job(cfg, task, robot, group, seed, steps, every, vendor, prepared, cost, layer_files, model_dir, record, tag):
    t = task
    tiers = None
    if group in TIERED:
        lf = layer_files[(t['id'], group)]
        tiers = dict(near_index_file=str(lf['abs']), near_index_sha256=lf['sha256'], w_target=cfg['layer_weights']['target'],
                     w_far=cfg['layer_weights']['far'], w_near=cfg['layer_weights']['near'], near_batch=cfg['near_batch'], layer=group)
    job = dict(id='%s_%s_%s_s%d' % (tag, group, t['short'], seed), worker='otdf_worker.py', task=t['id'], group=group,
                seed=seed, seed_origin=seed, steps=steps, eval_every=every, save_last=1 if model_dir else 0,
                model_dir=str(model_dir) if model_dir else None, state_dim=robot['state_dim'], action_dim=robot['action_dim'],
                xml=str(C.CODE/t['xml']), xml_sha256=t['xml_sha256'], env_module=robot['env_module'], env_class=robot['env_class'],
                score_min=t['score_min'], score_max=t['score_max'], otdf_config=t['otdf_config'], weight=True,
                proportion=PROPORTION[group], noreg=group == 'IQLCAL', reg_weight=t['reg_weight'], tiers=tiers,
                cost_file=str(cost[t['id']]['abs']), cost_sha256=cost[t['id']]['sha256'],
                expected_source_digest=prepared['source'][t['robot']]['digest'], expected_target_digest=prepared['target'][t['id']]['digest'],
                preflight_dirs=[], private_data=str(C.CODE/'data'), vendor=vendor, record_returns=record,
                device='cpu' if X.LOCAL else 'cuda')
    if X.LOCAL:
        job.update(X.local_cfg().get('otdf_job_overrides', {}))
    return job


def bc_job(cfg, task, robot, seed, steps, every, vendor, algo, model_dir, tag, target_rows, source_file, source_sha):
    t = task
    tp = asrc_data.target_path(t, C.load_tasks()[0], [], C.CODE/'data')       # SHA-256 checked inside
    job = dict(id='%s_BC_%s_s%d' % (tag, t['short'], seed), worker='bc_worker.py', task=t['id'], group='BC', learner='BC',
               seed=seed, seed_origin=seed, steps=steps, eval_every=every, state_dim=robot['state_dim'], action_dim=robot['action_dim'],
               target_rows=target_rows, score_min=t['score_min'], score_max=t['score_max'], xml=str(C.CODE/t['xml']),
               xml_sha256=t['xml_sha256'], target_dataset=str(tp), target_dataset_sha256=C.sha256_file(tp),
               source_dataset=str(source_file), source_dataset_sha256=source_sha,
               env_module=robot['env_module'], env_class=robot['env_class'], vendor=vendor, algo=algo, smoke=True,
               model_dir=str(model_dir), save_last=1)
    if X.LOCAL:
        job.update(X.local_cfg().get('bc_job_overrides', {}))
    return job


def raw_rows(path):
    import h5py
    with h5py.File(str(path), 'r') as f:
        return int(len(f['observations']))


# ----------------------------------------------------------------------------- main

def main(cfg):
    import numpy as np
    from asrc_support import Support
    import preflight_tests
    started = time.time()
    lc = X.local_cfg()
    res = dict(job=cfg['job'], started=C.utc(), checks={}, blocked=[], failures=[], notes=[])
    checks = res['checks']
    X.status(cfg, 'checks')
    checks['frozen_official'] = X.check_common(cfg)
    res['environment'] = environment_info()
    gpus = Q.visible_gpus(bool(lc))
    vendor = Q.setup_mujoco(bool(lc)) or (lc['vendor'] if lc else None)
    algo = lc['algo'] if lc else Q.transfer_learner()
    spec, tasks = C.load_tasks()
    task_list = [tasks[i] for i in cfg['tasks']]
    robots = spec['robots']

    # ---------------- 1. XML, data files, official arrays
    checks['xml'] = {}
    for t in task_list:
        got = C.sha256_file(C.CODE/t['xml'])
        assert got == t['xml_sha256'], 'XML hash mismatch: %s' % t['xml']
        checks['xml'][t['id']] = got
    d4rl = asrc_data.import_d4rl()
    private_data = C.CODE/'data'
    ot_dir = C.CODE/'ot_inputs'
    ot_dir.mkdir(exist_ok=True)
    prepared = dict(loader=dict(asrc_data_sha256=C.sha256_file(C.CODE/'asrc_data.py'),
                                d4rl_init_sha256=C.sha256_file(C.CODE/'d4rl'/'__init__.py'),
                                source='hdf5 -> dict (get_keys) -> d4rl.qlearning_dataset(None, dataset=dict), vendored d4rl 1.1',
                                target='call_tar_dataset_copy (official loop, next state = observations[i+1], N-1 rows)'),
                    source={}, target={})
    src_arrays, tar_arrays, segs = {}, {}, {}
    for robot in sorted({t['robot'] for t in task_list}):
        sp, sha = asrc_data.source_path(robot, spec)
        assert sp.is_file() and C.sha256_file(sp) == sha, 'source file missing or SHA-256 mismatch (%s)' % robot
        src, info = asrc_data.load_source_official(sp, d4rl)
        dg = asrc_data.dataset_digest(src)
        npy = ot_dir/('source_%s.npy' % robot)
        np.save(str(npy), asrc_data.features(src))
        prepared['source'][robot] = dict(file=C.rel(sp), sha256=sha, info=info, digest=dg, npy=C.rel(npy))
        src_arrays[robot] = src
        C.write_json(X.OUT/'progress.json', dict(time=C.utc(), phase='arrays', done='source %s' % robot))
    for t in task_list:
        tp = asrc_data.target_path(t, spec, [], private_data)
        tar = asrc_data.call_tar_dataset_copy(tp)
        seg, seg_info = asrc_data.segments_of_raw(tp)
        seg = seg[:len(tar['observations'])]
        dg = asrc_data.dataset_digest(tar)
        npy = ot_dir/('target_%s.npy' % t['short'])
        np.save(str(npy), asrc_data.features(tar))
        prepared['target'][t['id']] = dict(file=C.rel(tp), sha256=C.sha256_file(tp), raw_rows=raw_rows(tp), segments=seg_info,
                                           trajectories_in_training_rows=int(len(np.unique(seg))), digest=dg, npy=C.rel(npy))
        tar_arrays[t['id']], segs[t['id']] = tar, seg
    C.write_json(X.OUT/'PREPARED_MANIFEST.json', prepared)
    checks['data_files'] = 'passed'

    # ---------------- OT (background, CPU)
    py, versions, tried = find_ot_python(cfg)
    res['ot'] = dict(python=C.rel(py) if py else None, versions=versions, probes=tried)
    costdir = C.CODE/'costlogs'
    costdir.mkdir(exist_ok=True)
    pool = None
    if py is None:
        res['blocked'].append('OT: no Python with jax + ott-jax + jaxopt + h5py + numpy (job env and conda envs %s)' % cfg['ot']['conda_env_candidates'])
    else:
        reqs = []
        make_cost = C.CODE/'otdf_official'/'make_otdf_cost.py'
        for t in task_list:
            base = '%s-srcdatatype-medium-tardatatype-medium' % t['id']
            if (costdir/(base+'.hdf5')).exists():
                raise FileExistsError('cost file already exists: %s' % base)
            reqs.append(dict(task=t['id'], source_npy=str(ot_dir/('source_%s.npy' % t['robot'])),
                             target_npy=str(ot_dir/('target_%s.npy' % t['short'])),
                             source_sas_sha256=prepared['source'][t['robot']]['digest']['features_sas']['sha256'],
                             target_sas_sha256=prepared['target'][t['id']]['digest']['features_sas']['sha256'],
                             official_make_cost=str(make_cost), official_make_cost_sha256=C.OTDF_OFFICIAL_SHA256['otdf_official/make_otdf_cost.py'],
                             out_hdf5=str(costdir/(base+'.hdf5')), out_manifest=str(costdir/(base+'.manifest.json')),
                             request=str(costdir/(base+'.request.json')), log=str(costdir/(base+'.log'))))
        pool = OTPool(cfg, py, reqs)
        pool.start()
        res['ot']['parallel_processes'] = pool.parallel

    # ---------------- 2a. support statistics without OT values (CPU)
    X.status(cfg, 'support')
    sups = {}
    for t in task_list:
        sups[t['id']] = Support(tar_arrays[t['id']], segs[t['id']], src_arrays[t['robot']])
    C.write_json(X.OUT/'progress.json', dict(time=C.utc(), phase='support_without_cost', done=len(sups)))

    # ---------------- 3a. unit tests a, b, c on the GPU (stand-in OT values while OT runs)
    X.status(cfg, 'unit_tests')
    stamp = time.strftime('%Y%m%d_%H%M%S')
    base = C.CODE/('runs_'+stamp)

    def test_job(test, t, cost_entry=None):
        r = robots[t['robot']]
        j = dict(id='test_%s_%s' % (test, t['short']), worker='preflight_tests.py', test=test, task=t['id'],
                 state_dim=r['state_dim'], action_dim=r['action_dim'], otdf_config=t['otdf_config'], reg_weight=t['reg_weight'],
                 steps=cfg['unit_tests']['steps'], seed=cfg['smoke']['seed'], device='cpu' if lc else 'cuda', vendor=vendor,
                 preflight_dirs=[], private_data=str(private_data))
        if cost_entry:
            j.update(cost_file=str(cost_entry['abs']), cost_sha256=cost_entry['sha256'])
        if lc:
            j.update(lc.get('test_job_overrides', {}))
        return j
    tests = [test_job('a', tasks[i]) for i in cfg['unit_tests']['a_tasks']]
    tests += [test_job(x, t) for x in ('b', 'c') for t in task_list]
    done, failed, _ = Q.run_queue(cfg, tests, base/'tests_abc', gpus, 'tests_abc', bool(lc))
    unit = {}
    for j in tests:
        if j['id'] in done:
            unit[j['id']] = C.read_json(done[j['id']]/'COMPLETED.json')
        else:
            unit[j['id']] = dict(passed=False, error=failed.get(j['id']))
    res['unit_tests'] = unit

    # ---------------- wait for OT, verify, rebuild arrays
    cost = {}
    cost_manifest = dict(tasks={}, note='official solve_ot (AST), float16 / cosine / max_cost / Sinkhorn(1e-9, 100), chunks of 10000; only the empty tail chunk is skipped')
    if pool is not None:
        X.status(cfg, 'waiting_for_ot')
        while pool.is_alive():
            C.write_json(X.OUT/'progress.json', dict(time=C.utc(), phase='ot', done=sorted(pool.results), failed=sorted(pool.failed),
                                                     running=sorted(pool.procs)))
            pool.join(60)
        res['ot']['failed'] = pool.failed
        res['ot']['timed_out'] = pool.timed_out
        if pool.failed:
            res['blocked'].append('OT failed for %s' % sorted(pool.failed))
        total = 0
        for t in task_list:
            m = pool.results.get(t['id'])
            if m is None:
                continue
            path = costdir/m['cost_file']
            sha = C.sha256_file(path)
            ok = (sha == m['cost_sha256'] and m['source_rows'] == prepared['source'][t['robot']]['digest']['rows']
                  and m['source_sas_sha256'] == prepared['source'][t['robot']]['digest']['features_sas']['sha256']
                  and m['target_sas_sha256'] == prepared['target'][t['id']]['digest']['features_sas']['sha256'])
            assert ok, 'OT output of %s inconsistent with the exported arrays' % t['id']
            total += path.stat().st_size
            cost[t['id']] = dict(abs=path, sha256=sha)
            cost_manifest['tasks'][t['id']] = dict(cost_relpath=C.rel(path), cost_sha256=sha, cost_bytes=path.stat().st_size,
                                                   source_rows=m['source_rows'], target_rows=m['target_rows'],
                                                   source_sas_sha256=m['source_sas_sha256'], target_sas_sha256=m['target_sas_sha256'],
                                                   elapsed_seconds=m['elapsed_seconds'], packages=m['packages'], python=m['python'],
                                                   backend=m['backend'], settings=m['settings'], deviations=m['deviations'],
                                                   cost_min=m['cost_min'], cost_max=m['cost_max'])
        cost_manifest['total_bytes'] = total
        cost_manifest['ot_python'] = C.rel(py)
        cost_manifest['computed_on'] = 'server (j013), CPU'
    ot_ok = pool is not None and len(cost) == len(task_list) and not res['ot'].get('failed')
    res['ot']['status'] = 'done' if ot_ok else 'blocked'
    if ot_ok:
        # rebuild the official arrays and compare with the OT manifest
        rebuilt = {}
        for robot in sorted({t['robot'] for t in task_list}):
            sp, sha = asrc_data.source_path(robot, spec)
            s2, _ = asrc_data.load_source_official(sp, d4rl)
            rebuilt[robot] = asrc_data.dataset_digest(s2)['features_sas']['sha256']
        for t in task_list:
            tp = asrc_data.target_path(t, spec, [], private_data)
            t2 = asrc_data.dataset_digest(asrc_data.call_tar_dataset_copy(tp))['features_sas']['sha256']
            e = cost_manifest['tasks'][t['id']]
            assert rebuilt[t['robot']] == e['source_sas_sha256'] and t2 == e['target_sas_sha256'], 'rebuilt arrays differ (%s)' % t['id']
            costv = np.asarray(asrc_data.read_hdf5_dict(cost[t['id']]['abs'])['cost'])
            assert costv.shape == (prepared['source'][t['robot']]['digest']['rows'],) and np.isfinite(costv).all(), 'cost length / finiteness (%s)' % t['id']
            cost[t['id']]['values'] = costv
        checks['arrays_match_ot_manifest'] = 'passed'
        checks['cost_length_equals_source_rows'] = 'passed'
        C.write_json(X.OUT/'COST_MANIFEST.json', cost_manifest)
    else:
        # stand-in OT values (deterministic) so that the remaining steps still exercise the code; flagged everywhere
        for t in task_list:
            n = prepared['source'][t['robot']]['digest']['rows']
            path = costdir/('%s-STANDIN-NOT-FOR-J014.hdf5' % t['id'])
            if not path.exists():
                import h5py
                with h5py.File(str(path), 'w') as h:
                    h.create_dataset('cost', data=np.asarray(-np.random.RandomState(C.seed_of('standin', t['id'])).rand(n).astype(np.float16), np.float64), compression='gzip')
            cost[t['id']] = dict(abs=path, sha256=C.sha256_file(path), standin=True,
                                 values=np.asarray(asrc_data.read_hdf5_dict(path)['cost']))
        res['notes'].append('OT not available: support fractions by rho, test d, layer test, smoke and throughput used stand-in OT values (flagged); not valid for j014')

    # ---------------- 2b. support with OT values, layer files, test d
    support_out = {}
    layer_files = {}
    layer_dir = C.CODE/'layers'
    layer_dir.mkdir(exist_ok=True)
    for t in task_list:
        sup = sups[t['id']]
        sup.set_cost(cost[t['id']]['values'])
        s = sup.summary()
        s['ot_values'] = 'stand-in' if cost[t['id']].get('standin') else 'real'
        d = preflight_tests.random_tier_check(sup, t['id'])
        s['random_tier'] = d
        for g in TIERED:
            idx = sup.layers(g, t['id'])
            f = layer_dir/('%s_%s.npy' % (t['short'], g))
            np.save(str(f), idx)
            layer_files[(t['id'], g)] = dict(abs=f, sha256=C.array_digest(idx)['sha256'], rows=int(len(idx)))
            s.setdefault('layers', {})[g] = dict(rows=int(len(idx)), index_digest=C.array_digest(idx)['sha256'])
        support_out[t['id']] = s
        unit['test_d_%s' % t['short']] = dict(d, test='d', task=t['id'], cost_kind=s['ot_values'])
    C.write_json(X.OUT/'SUPPORT.json', dict(tasks=support_out, definition=C.read_json(C.CODE/'config.json')['support_definition']))

    # layer test on the GPU with the (real or stand-in) OT values
    lt = [test_job('layer', tasks[i], cost[i]) for i in cfg['unit_tests']['layer_tasks']]
    done, failed, _ = Q.run_queue(cfg, lt, base/'tests_layer', gpus, 'tests_layer', bool(lc))
    for j in lt:
        unit[j['id']] = C.read_json(done[j['id']]/'COMPLETED.json') if j['id'] in done else dict(passed=False, error=failed.get(j['id']))
    tests_ok = all(v.get('passed') for v in unit.values())
    checks['unit_tests'] = 'passed' if tests_ok else 'FAILED'
    if not tests_ok:
        res['failures'].append('unit tests: %s' % sorted(k for k, v in unit.items() if not v.get('passed')))

    # ---------------- 4. smoke
    X.status(cfg, 'smoke')
    seed = cfg['smoke']['seed']
    smoke = []
    smoke_models = C.CODE/'smoke_models'
    for t in task_list:
        r = robots[t['robot']]
        for g in OTDF_GROUPS:
            smoke.append(otdf_job(cfg, t, r, g, seed, cfg['smoke']['steps'], cfg['smoke']['steps'], vendor, prepared, cost,
                                  layer_files, smoke_models/('%s_%s' % (t['short'], g)), False, 'smoke'))
        sp, ssha = asrc_data.source_path(t['robot'], spec)
        smoke.append(bc_job(cfg, t, r, seed, cfg['smoke']['steps'], cfg['smoke']['steps'], vendor, algo,
                            smoke_models/('%s_BC' % t['short']), 'smoke', prepared['target'][t['id']]['raw_rows'], sp, ssha))
        if t['id'] in cfg['smoke']['iqlcal_tasks']:
            smoke.append(otdf_job(cfg, t, r, 'IQLCAL', seed, cfg['smoke']['steps'], cfg['smoke']['steps'], vendor, prepared, cost,
                                  layer_files, smoke_models/('%s_IQLCAL' % t['short']), False, 'smoke'))
    done, failed, _ = Q.run_queue(cfg, smoke, base/'smoke', gpus, 'smoke', bool(lc))
    smoke_res = {}
    import torch
    for j in smoke:
        if j['id'] not in done:
            smoke_res[j['id']] = dict(passed=False, error=failed.get(j['id']))
            continue
        c = C.read_json(done[j['id']]/'COMPLETED.json')
        saved = c.get('saved_models') or []
        ok = bool(c.get('finite')) and c.get('evaluations') == 1 and len(saved) == 1
        if ok:
            f = Path(j['model_dir'])/(saved[0].get('file') or Path(saved[0]['path']).name)
            ok = f.is_file() and C.sha256_file(f) == saved[0]['sha256']
            blob = torch.load(str(f), map_location='cpu')
            sdict = blob['policy'] if 'policy' in blob else blob
            ok = ok and all(bool(torch.isfinite(v).all()) for v in sdict.values())
        smoke_res[j['id']] = dict(passed=bool(ok), group=j['group'], task=j['task'], finite=c.get('finite'),
                                  evaluations=c.get('evaluations'), model_saved=len(saved) == 1,
                                  near_rows=(c.get('tiers') or {}).get('near_rows'), tier_stats=c.get('tier_stats'),
                                  wall_seconds=c.get('wall_seconds'), train_steps_per_second=c.get('train_steps_per_second'))
    res['smoke'] = smoke_res
    smoke_ok = all(v['passed'] for v in smoke_res.values()) and len(smoke_res) == len(smoke)
    checks['smoke'] = 'passed' if smoke_ok else 'FAILED'
    if not smoke_ok:
        res['failures'].append('smoke: %s' % sorted(k for k, v in smoke_res.items() if not v['passed']))

    # ---------------- 5. throughput
    X.status(cfg, 'throughput')
    thr = {}
    for tid in cfg['throughput']['tasks']:
        t = tasks[tid]
        r = robots[t['robot']]
        jobs = [otdf_job(cfg, t, r, g, seed, cfg['throughput']['steps'], cfg['throughput']['eval_every'], vendor, prepared, cost,
                         layer_files, None, False, 'thr') for g in OTDF_GROUPS]
        t0 = time.time()
        done, failed, _ = Q.run_queue(dict(cfg, launch_gap_seconds=0), jobs, base/('throughput_'+t['short']), gpus, 'throughput', bool(lc))
        wall = time.time()-t0
        runs = {}
        for j in jobs:
            if j['id'] in done:
                c = C.read_json(done[j['id']]/'COMPLETED.json')
                runs[j['group']] = dict(train_steps_per_second=c['train_steps_per_second'], train_seconds=c['train_seconds'],
                                        steps=j['steps'], load_seconds=c['load_seconds'],
                                        vae_seconds=c['vae_seconds'], eval_seconds=c['eval_seconds'], eval_env_steps=c['eval_env_steps'],
                                        gpu_peak_allocated_bytes=c['gpu_peak_allocated_bytes'], wall_seconds=c['wall_seconds'])
        thr[tid] = dict(processes=len(jobs), completed=len(runs), wall_seconds=wall, runs=runs, failed=failed)
    res['throughput'] = estimate(cfg, thr, smoke_res)
    thr_ok = all(v['completed'] == v['processes'] for v in thr.values())
    checks['throughput'] = 'passed' if thr_ok else 'FAILED'
    if not thr_ok:
        res['failures'].append('throughput runs failed')

    # ---------------- 6. normalization table, state restoration
    res['normalization'] = normalization_table(cfg, task_list, spec)
    res['state_restoration'] = restoration_check(task_list, robots, tar_arrays, bool(lc))
    rest_ok = all(v['passed'] for v in res['state_restoration'].values())
    checks['state_restoration'] = 'passed' if rest_ok else 'FAILED'
    if not rest_ok:
        res['failures'].append('state restoration')

    res['support'] = {k: dict(tau=v['tau'], tau_state=v['tau_state'], near_all_fraction=v.get('near_all_fraction'),
                              by_rho=v.get('by_rho'), ot_values=v['ot_values']) for k, v in support_out.items()}
    res['cost_files'] = dict(total_bytes=cost_manifest.get('total_bytes'), status=res['ot']['status'])
    res['all_passed'] = bool(ot_ok and tests_ok and smoke_ok and thr_ok and rest_ok and not res['blocked'] and not res['failures'])
    res['elapsed_hours'] = (time.time()-started)/3600
    C.write_json(X.OUT/'PREFLIGHT.json', res)
    report(res, task_list)
    X.status(cfg, 'completed' if res['all_passed'] else 'finished_with_problems', all_passed=res['all_passed'])
    print((X.OUT/'REPORT_ZH.md').read_text(encoding='utf-8'), flush=True)
    return 0 if res['all_passed'] else 1


def estimate(cfg, thr, smoke_res):
    """T for 1M OTDF steps with 6 concurrent processes and one j014 job (BC first, then the 6th OTDF run)."""
    out = dict(measured=thr, assumptions=dict(otdf_steps=1000000, evals=100, episodes=10, max_episode_steps=1000,
                                              bc_steps=200000, bc_evals=40))
    worst = None
    for tid, v in thr.items():
        if not v['runs']:
            continue
        est = {}
        for g, r in v['runs'].items():
            its = r['train_steps_per_second'] or (r['steps']/r['train_seconds'] if r.get('train_seconds') else 0.)
            per_env_step = r['eval_seconds']/max(r['eval_env_steps'], 1)
            train_h = 1e6/its/3600 if its > 0 else None
            upper_eval_h = 100*10*1000*per_env_step/3600
            measured_eval_h = 100*r['eval_seconds']/2/3600
            est[g] = dict(train_hours=train_h, eval_hours_upper=upper_eval_h, eval_hours_at_measured_length=measured_eval_h,
                          T_hours_upper=(train_h or 0)+upper_eval_h+(r['load_seconds']+r['vae_seconds'])/3600,
                          T_hours_typical=(train_h or 0)+measured_eval_h+(r['load_seconds']+r['vae_seconds'])/3600)
        tmax = max(e['T_hours_upper'] for e in est.values())
        ttyp = max(e['T_hours_typical'] for e in est.values())
        bc = [s.get('wall_seconds') for k, s in smoke_res.items() if s.get('group') == 'BC' and s.get('wall_seconds')]
        bc_h = (max(bc)/cfg['smoke']['steps']*200000/3600) if bc else 1.
        out[tid] = dict(per_group=est, T_otdf_hours_upper=tmax, T_otdf_hours_typical=ttyp, bc_hours_estimate=bc_h,
                        job_hours_upper=tmax+bc_h, job_hours_typical=ttyp+bc_h)
        worst = max(worst or 0., tmax+bc_h)
    out['job_hours_upper_worst'] = worst
    if worst is None:
        out['recommendation'] = 'no estimate (throughput runs failed)'
    elif worst <= 22:
        out['recommendation'] = 'no split: one job per (task, seed); max_hours = min(24, ceil(1.3 x %.1f))' % worst
    else:
        out['recommendation'] = ('split every (task, seed) into two jobs that both contain rho80: A = {rho80, rho0, rho20T, BC}, '
                                 'B = {rho80, rho100T, rho80T, rho80R} (build_jobs.py --j014-layout split)')
    return out


def normalization_table(cfg, task_list, spec):
    p = C.CODE/'otdf_official'/'info.py'
    s = importlib.util.spec_from_file_location('otdf_info', str(p))
    info = importlib.util.module_from_spec(s)
    s.loader.exec_module(info)
    grid = cfg['normalization_grid']
    out = {}
    for t in task_list:
        if t['odrl_key']:
            lo1, hi1, src1 = t['score_min'], t['score_max'], 'ODRL envs/infos.py: %s' % t['odrl_key']
        else:
            r = spec['odrl_reference_for_hopper_family']
            lo1, hi1, src1 = r['score_min'], r['score_max'], 'ODRL hopper family reference (no hopper-morph entry in ODRL)'
        lo2, hi2 = info.REF_MIN_SCORE[t['otdf_key']], info.REF_MAX_SCORE[t['otdf_key']]
        rows = [dict(raw=x, odrl=100*(x-lo1)/(hi1-lo1), otdf=info.get_normalized_score(t['otdf_key'], x),
                     diff=info.get_normalized_score(t['otdf_key'], x)-100*(x-lo1)/(hi1-lo1)) for x in grid]
        out[t['id']] = dict(odrl=dict(min=lo1, max=hi1, source=src1), otdf=dict(key=t['otdf_key'], min=lo2, max=hi2),
                            primary=t['norm_primary'], rows=rows)
    return out


def restoration_check(task_list, robots, tar_arrays, local):
    from simcheck_worker import restore
    out = {}
    for t in task_list:
        r = robots[t['robot']]
        if local:
            import importlib as il
            mod = il.import_module(X.local_cfg()['env_module_name'])
            env = mod.make(r['state_dim'], r['action_dim'])
        else:
            import gym
            constructor = getattr(__import__(r['env_module'], fromlist=[r['env_class']]), r['env_class'])
            env = gym.wrappers.TimeLimit(constructor(xml_file=str(C.CODE/t['xml'])), max_episode_steps=1000)
        obs = tar_arrays[t['id']]['observations']
        errs = []
        for i in (0, 100, 500, 1000, 2000):
            if i < len(obs):
                e, _ = restore(env, obs[i], r['nq'], r['nv'])
                errs.append(e)
        env.close()
        out[t['id']] = dict(max_abs_error=max(errs), checked=len(errs), passed=max(errs) <= 1e-6)
    return out


def report(res, task_list):
    L = ['# j013_adapt_preflight：数据、OT 代价、单元测试、冒烟与吞吐预检', '',
         '全部通过：%s。冒烟与预检不读取、不报告回报。' % ('是' if res['all_passed'] else '否'), '']
    if res['blocked']:
        L += ['## 受阻', ''] + ['- %s' % b for b in res['blocked']] + ['']
    if res['failures']:
        L += ['## 未通过', ''] + ['- %s' % f for f in res['failures']] + ['']
    L += ['## 检查', ''] + ['- %s：%s' % (k, v if isinstance(v, str) else '见 PREFLIGHT.json') for k, v in res['checks'].items()] + ['']
    ot = res['ot']
    L += ['## OT 代价（服务器端计算）', '', '- 状态：%s；OT Python：%s；版本：%s' % (ot.get('status'), ot.get('python'), ot.get('versions')),
          '- 并行进程：%s；合计大小：%s 字节' % (ot.get('parallel_processes'), res['cost_files'].get('total_bytes')), '']
    L += ['## 支撑统计（每个任务）', '', '|任务|tau|tau_state|近邻层/全部源（rho=1）|rho=0.2 占源/占D_ret|rho=0.8 占源/占D_ret|OT 值|',
          '|---|---|---|---|---|---|---|']
    for t in task_list:
        s = res['support'][t['id']]
        br = s.get('by_rho') or {}
        f = lambda k: '%.4f / %.4f' % (br[k]['near_fraction_of_source'], br[k]['near_fraction_of_ret']) if k in br else ''
        L.append('|%s|%.3f|%.3f|%s|%s|%s|%s|' % (t['id'], s['tau'], s['tau_state'],
                 '%.4f' % s['near_all_fraction'] if s.get('near_all_fraction') is not None else '', f('0.2'), f('0.8'), s['ot_values']))
    L += ['', '## 单元测试', '']
    for k, v in sorted(res['unit_tests'].items()):
        L.append('- %s：%s%s' % (k, '通过' if v.get('passed') else '未通过', '（OT 值：%s）' % v.get('cost_kind') if v.get('cost_kind') else ''))
    sm = res['smoke']
    L += ['', '## 冒烟', '', '共 %d 项，通过 %d 项。' % (len(sm), sum(1 for v in sm.values() if v['passed'])), '']
    th = res['throughput']
    L += ['## 吞吐与 j014 拆分建议', '']
    for tid, v in th.items():
        if isinstance(v, dict) and 'per_group' in v:
            L.append('- %s：6 进程同卡，100 万步 OTDF 估计 %.1f 小时（上界，评估按满长回合）/ %.1f 小时（按实测回合长度）；一个 j014 任务估计 %.1f / %.1f 小时' % (
                tid, v['T_otdf_hours_upper'], v['T_otdf_hours_typical'], v['job_hours_upper'], v['job_hours_typical']))
    L += ['- 建议：%s' % th.get('recommendation'), '']
    L += ['## 归一化对照（固定的原始回报网格，非实验回报）', '']
    for tid, v in res['normalization'].items():
        L.append('- %s：ODRL [%g, %g]（%s） vs OTDF info.py %s [%g, %g]；主口径：%s' % (
            tid, v['odrl']['min'], v['odrl']['max'], v['odrl']['source'], v['otdf']['key'], v['otdf']['min'], v['otdf']['max'], v['primary']))
        L.append('  ' + '；'.join('原始 %g → %.2f / %.2f' % (r['raw'], r['odrl'], r['otdf']) for r in v['rows']))
    L += ['', '## 仿真器状态恢复检查（j016 前提）', '']
    for tid, v in res['state_restoration'].items():
        L.append('- %s：最大误差 %.2e，%s' % (tid, v['max_abs_error'], '通过' if v['passed'] else '未通过'))
    for n in res['notes']:
        L.append('- 注：%s' % n)
    L += ['', '总耗时 %.2f 小时。' % res['elapsed_hours']]
    (X.OUT/'REPORT_ZH.md').write_text('\n'.join(L)+'\n', encoding='utf-8')
