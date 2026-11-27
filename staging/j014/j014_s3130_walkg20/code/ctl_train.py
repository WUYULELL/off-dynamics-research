"""j014 / j014c controller: OTDF groups and BC on one task and seed (j014_s<seed>_<task>) or the IQL calibration
(j014c_iqlcal: OTDF proportion 0 + noreg on two hopper tasks x 5 seeds). All groups of a job run on the same GPU in
the same period (at most max_per_gpu processes; BC first, it is short).

Before training the controller (1) finds the completed preflight job (PREFLIGHT.json all_passed) and reads its
COST_MANIFEST / SUPPORT / PREPARED manifests, (2) rebuilds the official training arrays and requires their
[s, a, s'] SHA-256 and row counts to equal the manifests, (3) checks the cost file hash and length, (4) rebuilds
the support statistics and layers and requires tau, tau_state and the near-layer sizes / index digests to equal
SUPPORT.json, then writes the layer index files used by the workers.
Outputs: summary.json (Last5 per group), RUNS.csv, CURVES.json, MODEL_MANIFEST.json (paths and SHA-256 only, no
returns; read by j015b / j016), STATUS.json (no returns), REPORT_ZH.md (descriptive, no verdict).
"""
import csv
from pathlib import Path
import time

import asrc_common as C
import asrc_ctl as X
import asrc_data
import asrc_queue as Q

TIERED = ('rho20T', 'rho100T', 'rho80T', 'rho80R')
NAMES = {'rho0': 'ρ0（OTDF，proportion 0）', 'rho80': 'ρ80（OTDF 官方原样，proportion 0.8）',
         'rho20T': 'ρ20T（proportion 0.2 + 分层）', 'rho100T': 'ρ100T（proportion 1.0 + 分层）',
         'rho80T': 'ρ80T（proportion 0.8 + 分层）', 'rho80R': 'ρ80R（proportion 0.8 + 随机分层对照）',
         'BC': 'BC（只用目标数据，j011 worker）', 'IQLCAL': 'IQL（OTDF proportion 0 + noreg）'}


def main(cfg):
    import numpy as np
    from asrc_support import Support
    started = time.time()
    lc = X.local_cfg()
    X.status(cfg, 'prepare')
    X.check_common(cfg)
    gpus = Q.visible_gpus(bool(lc))
    vendor = Q.setup_mujoco(bool(lc)) or (lc['vendor'] if lc else None)
    algo = lc['algo'] if lc else Q.transfer_learner()
    spec, tasks = C.load_tasks()
    pre = X.resolve_preflight(cfg)
    cost_m, support_m, prepared_m = X.preflight_files(pre)
    d4rl = asrc_data.import_d4rl()
    private_data = C.CODE/'data'
    layer_dir = C.CODE/'layers'
    layer_dir.mkdir(exist_ok=True)
    checks, layer_files, expected = {}, {}, {}
    for tid in cfg['tasks']:
        t = tasks[tid]
        src, tar, info = asrc_data.load_task_arrays(t, spec, d4rl, [pre], private_data)
        sdig, tdig = asrc_data.dataset_digest(src), asrc_data.dataset_digest(tar)
        cm = cost_m['tasks'][tid]
        assert sdig['features_sas']['sha256'] == cm['source_sas_sha256'] and sdig['rows'] == cm['source_rows'], 'source array != preflight (%s)' % tid
        assert tdig['features_sas']['sha256'] == cm['target_sas_sha256'] and tdig['rows'] == cm['target_rows'], 'target array != preflight (%s)' % tid
        cpath, centry = X.cost_file_of(pre, cost_m, tid)
        cost = np.asarray(asrc_data.read_hdf5_dict(cpath)['cost'])
        assert cost.shape == (sdig['rows'],) and np.isfinite(cost).all(), 'cost length / finiteness (%s)' % tid
        expected[tid] = dict(source=sdig, target=tdig, cost_file=cpath, cost_sha256=centry['cost_sha256'],
                             target_rows_raw=prepared_m['target'][tid]['raw_rows'])
        need_layers = any(g['name'] in TIERED for g in cfg['groups'])
        if need_layers:
            tp = asrc_data.target_path(t, spec, [pre], private_data)
            seg = asrc_data.segments_of_raw(tp)[0][:len(tar['observations'])]
            sup = Support(tar, seg, src, cost)
            sm = support_m['tasks'][tid]
            assert abs(sup.tau-sm['tau']) <= 1e-9*max(1., abs(sm['tau'])), 'tau != SUPPORT.json (%s)' % tid
            assert abs(sup.tau_state-sm['tau_state']) <= 1e-9*max(1., abs(sm['tau_state'])), 'tau_state != SUPPORT.json (%s)' % tid
            for g in cfg['groups']:
                if g['name'] not in TIERED:
                    continue
                idx = sup.layers(g['name'], tid)
                dig = C.array_digest(idx)['sha256']
                assert len(idx) == sm['layers'][g['name']]['rows'] and dig == sm['layers'][g['name']]['index_digest'], \
                    'layer %s of %s differs from SUPPORT.json' % (g['name'], tid)
                f = layer_dir/('%s_%s.npy' % (t['short'], g['name']))
                np.save(str(f), idx)
                layer_files[(tid, g['name'])] = dict(abs=f, sha256=dig, rows=int(len(idx)))
        checks[tid] = dict(arrays='match preflight', cost_sha256=centry['cost_sha256'],
                           layers={g: v['rows'] for (task_id, g), v in layer_files.items() if task_id == tid})
        del src, tar
    jobs = plan(cfg, spec, tasks, expected, layer_files, vendor, algo, pre, private_data)
    C.write_json(X.OUT/'PLAN.json', [dict(id=j['id'], task=j['task'], group=j['group'], seed=j['seed'], steps=j['steps']) for j in jobs])
    X.status(cfg, 'main', runs=len(jobs), preflight=C.rel(pre))
    stamp = time.strftime('%Y%m%d_%H%M%S')
    done, failed, events = Q.run_queue(cfg, jobs, C.CODE/('runs_'+stamp)/'main', gpus, 'main', bool(lc))
    if failed:
        X.status(cfg, 'main_failed', failed=failed, completed=len(done))
        return 1
    summarize(cfg, jobs, done, checks, started, pre)
    X.status(cfg, 'completed', runs=len(jobs))
    print((X.OUT/'REPORT_ZH.md').read_text(encoding='utf-8'), flush=True)
    return 0


def plan(cfg, spec, tasks, expected, layer_files, vendor, algo, pre, private_data):
    lc = X.local_cfg()
    jobs = []
    order = sorted(cfg['groups'], key=lambda g: 0 if g['learner'] == 'BC' else 1)     # BC first (short)
    for tid in cfg['tasks']:
        t = tasks[tid]
        r = spec['robots'][t['robot']]
        for g in order:
            for seed in cfg['seeds']:
                name = '%s_%s_s%d' % (g['name'], t['short'], seed)
                model_dir = C.CODE/'outputs'/'models'/('%s_%s_s%d' % (t['short'], g['name'], seed))
                if g['learner'] == 'BC':
                    sp, ssha = asrc_data.source_path(t['robot'], spec)
                    tp = asrc_data.target_path(t, spec, [pre], private_data)        # SHA-256 checked inside
                    job = dict(id=name, worker='bc_worker.py', task=tid, group='BC', learner='BC', seed=seed,
                               seed_origin=cfg['seed_origin'], steps=cfg['bc']['steps'], eval_every=cfg['bc']['eval_every'],
                               state_dim=r['state_dim'], action_dim=r['action_dim'], target_rows=expected[tid]['target_rows_raw'],
                               score_min=t['score_min'], score_max=t['score_max'], xml=str(C.CODE/t['xml']), xml_sha256=t['xml_sha256'],
                               target_dataset=str(tp), target_dataset_sha256=C.sha256_file(tp),
                               source_dataset=str(sp), source_dataset_sha256=ssha, env_module=r['env_module'], env_class=r['env_class'],
                               vendor=vendor, algo=algo, smoke=False, model_dir=str(model_dir), save_last=cfg['bc']['save_last'])
                    if lc:
                        job.update(lc.get('bc_job_overrides', {}))
                else:
                    tiers = None
                    if g['name'] in TIERED:
                        lf = layer_files[(tid, g['name'])]
                        tiers = dict(near_index_file=str(lf['abs']), near_index_sha256=lf['sha256'], w_target=cfg['layer_weights']['target'],
                                     w_far=cfg['layer_weights']['far'], w_near=cfg['layer_weights']['near'], near_batch=cfg['near_batch'],
                                     layer=g['name'])
                    job = dict(id=name, worker='otdf_worker.py', task=tid, group=g['name'], seed=seed, seed_origin=cfg['seed_origin'],
                               steps=cfg['otdf']['steps'], eval_every=cfg['otdf']['eval_every'], save_last=cfg['otdf']['save_last'],
                               model_dir=str(model_dir), state_dim=r['state_dim'], action_dim=r['action_dim'],
                               xml=str(C.CODE/t['xml']), xml_sha256=t['xml_sha256'], env_module=r['env_module'], env_class=r['env_class'],
                               score_min=t['score_min'], score_max=t['score_max'], otdf_config=t['otdf_config'], weight=True,
                               proportion=g['proportion'], noreg=bool(g.get('noreg', False)), reg_weight=t['reg_weight'], tiers=tiers,
                               cost_file=str(expected[tid]['cost_file']), cost_sha256=expected[tid]['cost_sha256'],
                               expected_source_digest=expected[tid]['source'], expected_target_digest=expected[tid]['target'],
                               preflight_dirs=[str(pre)], private_data=str(private_data), vendor=vendor, record_returns=True,
                               device='cpu' if lc else 'cuda')
                    if lc:
                        job.update(lc.get('otdf_job_overrides', {}))
                jobs.append(job)
    return jobs


def summarize(cfg, jobs, done, checks, started, pre):
    runs, table, models = [], {}, {}
    for j in jobs:
        r = Q.collect_evals(done[j['id']], j['steps'], j['eval_every'], j['seed'], j['seed_origin'])
        c = C.read_json(done[j['id']]/'COMPLETED.json')
        table[(j['task'], j['group'], j['seed'])] = r
        runs.append(dict(task=j['task'], group=j['group'], seed=j['seed'], **{k: v for k, v in r.items() if k != 'curve'}))
        entries = []
        for m in c.get('saved_models') or []:
            path = C.CODE/m['path'] if 'path' in m else Path(j['model_dir'])/m['file']
            assert C.sha256_file(path) == m['sha256'], 'saved model changed: %s' % path.name
            entries.append(dict(step=m['step'], path=C.rel(path), sha256=m['sha256'], bytes=m['bytes'],
                                kind='bc_policy' if j['group'] == 'BC' else 'otdf_actor'))
        assert len(entries) == (cfg['bc']['save_last'] if j['group'] == 'BC' else cfg['otdf']['save_last']), 'missing saved models (%s)' % j['id']
        models['%s|%s|%d' % (j['task'], j['group'], j['seed'])] = dict(task=j['task'], group=j['group'], seed=j['seed'], models=entries)
    C.write_json(X.OUT/'MODEL_MANIFEST.json', dict(job=cfg['job'], note='paths relative to this job\'s code/ folder; no returns',
                                                   entries=models))
    tasks = {}
    groups = [g['name'] for g in cfg['groups']]
    for tid in cfg['tasks']:
        per = {}
        for g in groups:
            v = [table[(tid, g, s)]['last5'] for s in cfg['seeds']]
            st = C.stats(v)
            per[g] = dict(per_seed={str(s): table[(tid, g, s)]['last5'] for s in cfg['seeds']}, mean=st['mean'], sd=st['sd'], se=st['se'],
                          final={str(s): table[(tid, g, s)]['final'] for s in cfg['seeds']},
                          length_mean=sum(table[(tid, g, s)]['length_mean'] for s in cfg['seeds'])/len(cfg['seeds']),
                          fraction_under_200=sum(table[(tid, g, s)]['fraction_under_200'] for s in cfg['seeds'])/len(cfg['seeds']))
        tasks[tid] = per
    C.write_json(X.OUT/'summary.json', dict(job=cfg['job'], seeds=cfg['seeds'], groups=groups, tasks=tasks, checks=checks,
                                            preflight=C.rel(pre), metric='Last5 = mean of the last five evaluation points '
                                            '(OTDF: 960k-1M every 10k; BC: 180k-200k every 5k), 10 episodes each, normalised as tasks.json score_min/max',
                                            elapsed_hours=(time.time()-started)/3600))
    with (X.OUT/'RUNS.csv').open('w', newline='', encoding='utf-8') as f:
        keys = ['task', 'group', 'seed', 'last5', 'final', 'best_point', 'length_mean', 'length_min', 'fraction_under_200', 'last5_raw_mean']
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in runs:
            w.writerow(r)
    C.write_json(X.OUT/'CURVES.json', {j['id']: table[(j['task'], j['group'], j['seed'])]['curve'] for j in jobs})
    L = ['# %s（描述性，不判定；判定由 analysis_tools/j014_matrix.py 在全部任务结束后按 PLAN 规则做）' % cfg['job'], '',
         '种子 %s；主指标 Last5。' % cfg['seeds'], '']
    for tid, per in tasks.items():
        L += ['## %s' % tid, '', '|组|Last5 均值|SD|各种子|回合长度|<200 步比例|', '|---|---|---|---|---|---|']
        for g in groups:
            r = per[g]
            L.append('|%s|%.2f|%.2f|%s|%.1f|%.2f|' % (NAMES.get(g, g), r['mean'], r['sd'] or 0., ' / '.join('%.1f' % x for x in r['per_seed'].values()),
                                                      r['length_mean'], r['fraction_under_200']))
        L.append('')
    L.append('分层：%s' % {k: v['layers'] for k, v in checks.items()})
    L.append('总耗时 %.2f 小时。' % ((time.time()-started)/3600))
    (X.OUT/'REPORT_ZH.md').write_text('\n'.join(L)+'\n', encoding='utf-8')
