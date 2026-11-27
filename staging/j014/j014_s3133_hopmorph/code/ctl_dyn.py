"""j015a_dyn_models controller: probabilistic dynamics ensembles (K = 5) per (task, seed) and H* calibration.

Phase 'source': one worker per (robot, seed) trains the K source-stage members (shared by the robot's tasks).
Phase 'target': one worker per (task, seed) runs the cross calibration and the final fine-tuning.
Outputs: H_STAR.json (H*(eps0) per task and seed), ERROR_CURVES.csv (q50 / q90 / count per horizon),
DYN_MANIFEST.json (final members, paths relative to code/ and SHA-256), STATUS.json, REPORT_ZH.md.
Does not need the preflight job (hopper-morph target data are part of this job's code/otdf_official/datasets).
"""
import csv
import time

import asrc_common as C
import asrc_ctl as X
import asrc_data
import asrc_queue as Q


def main(cfg):
    started = time.time()
    lc = X.local_cfg()
    X.status(cfg, 'prepare')
    X.check_common(cfg)
    gpus = Q.visible_gpus(bool(lc))
    spec, tasks = C.load_tasks()
    private_data = C.CODE/'data'
    m = cfg['model']
    stamp = time.strftime('%Y%m%d_%H%M%S')
    base = C.CODE/('runs_'+stamp)
    robots = sorted({tasks[t]['robot'] for t in cfg['tasks']})
    src_jobs = []
    for robot in robots:
        r = spec['robots'][robot]
        sp, ssha = asrc_data.source_path(robot, spec)
        for seed in cfg['seeds']:
            j = dict(id='source_%s_s%d' % (robot, seed), worker='dyn_worker.py', kind='source', robot=robot, seed=seed,
                     state_dim=r['state_dim'], action_dim=r['action_dim'], source_dataset=str(sp), source_sha256=ssha,
                     model=m, source_steps=m['source']['steps'], source_eval_every=m['source']['eval_every'],
                     device='cpu' if lc else 'cuda')
            if lc:
                j.update(lc.get('dyn_source_overrides', {}))
            src_jobs.append(j)
    X.status(cfg, 'source', runs=len(src_jobs))
    done_s, failed, _ = Q.run_queue(cfg, src_jobs, base/'source', gpus, 'source', bool(lc))
    if failed:
        X.status(cfg, 'source_failed', failed=failed)
        return 1
    tgt_jobs = []
    for tid in cfg['tasks']:
        t = tasks[tid]
        r = spec['robots'][t['robot']]
        tp = asrc_data.target_path(t, spec, [], private_data)
        for seed in cfg['seeds']:
            j = dict(id='target_%s_s%d' % (t['short'], seed), worker='dyn_worker.py', kind='target', task=tid, seed=seed,
                     robot=t['robot'], state_dim=r['state_dim'], action_dim=r['action_dim'], vel_start=r['vel_start'],
                     target_dataset=str(tp), target_sha256=C.sha256_file(tp), model=m,
                     source_dir=str(done_s['source_%s_s%d' % (t['robot'], seed)]),
                     target_max_steps=m['target']['max_steps'], target_eval_every=m['target']['eval_every'],
                     model_dir=str(C.CODE/'outputs'/'models'/('%s_s%d' % (t['short'], seed))), device='cpu' if lc else 'cuda')
            if lc:
                j.update(lc.get('dyn_target_overrides', {}))
            tgt_jobs.append(j)
    X.status(cfg, 'target', runs=len(tgt_jobs))
    done_t, failed, _ = Q.run_queue(cfg, tgt_jobs, base/'target', gpus, 'target', bool(lc))
    if failed:
        X.status(cfg, 'target_failed', failed=failed)
        return 1
    hstar, manifest, curves_rows, fits = {}, {}, [], {}
    for j in tgt_jobs:
        c = C.read_json(done_t[j['id']]/'COMPLETED.json')
        key = '%s|%d' % (j['task'], j['seed'])
        hstar[key] = dict(task=j['task'], seed=j['seed'], h_star=c['h_star'], h_max=c['h_max'], trajectories=c['trajectories'],
                          folds=len(c['folds']))
        for e in c['saved_models']:
            assert C.sha256_file(C.CODE/e['path']) == e['sha256']
        manifest[key] = dict(task=j['task'], seed=j['seed'], members=c['saved_models'])
        for h, (q50, q90, n) in enumerate(zip(c['q50'], c['q90'], c['counts']), start=1):
            curves_rows.append(dict(task=j['task'], seed=j['seed'], h=h, q50=q50, q90=q90, n=n))
        fits[key] = dict(folds=[dict(fold=f['fold'], held_out=f['held_out_trajectories'],
                                     validation_ft=[x['best_validation'] for x in f['fits']],
                                     validation_source_only=[x['source_only_validation'] for x in f['fits']]) for f in c['folds']],
                         final=[dict(member=x['member'], best_validation=x['best_validation'], best_step=x['best_step'],
                                     source_only=x['source_only_validation']) for x in c['final']['fits']])
    C.write_json(X.OUT/'H_STAR.json', dict(eps0=m['eps0'], quantile=m['quantile'], entries=hstar))
    C.write_json(X.OUT/'DYN_MANIFEST.json', dict(note='final members fine-tuned on all target data (10% for early stopping); paths relative to code/', entries=manifest))
    C.write_json(X.OUT/'FITS.json', fits)
    with (X.OUT/'ERROR_CURVES.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['task', 'seed', 'h', 'q50', 'q90', 'n'])
        w.writeheader()
        for r in curves_rows:
            w.writerow(r)
    L = ['# j015a_dyn_models：概率动力学集成与可信步长 H*', '',
         'K = %d，网络 %s SiLU，高斯负对数似然；源数据 %d 步 / 目标微调至多 %d 步；按轨迹 min(轨迹数, 5) 折交叉校准，0.9 分位数。' % (
             m['members'], m['hidden'], m['source']['steps'], m['target']['max_steps']), '',
         '|任务|种子|轨迹数|h_max|' + '|'.join('H*(%g)' % e for e in m['eps0']) + '|', '|---|---|---|---|' + '---|'*len(m['eps0'])]
    for key in sorted(hstar):
        v = hstar[key]
        L.append('|%s|%d|%d|%d|%s|' % (v['task'], v['seed'], v['trajectories'], v['h_max'],
                                       '|'.join(str(v['h_star']['%g' % e]) for e in m['eps0'])))
    L += ['', '总耗时 %.2f 小时。' % ((time.time()-started)/3600)]
    (X.OUT/'REPORT_ZH.md').write_text('\n'.join(L)+'\n', encoding='utf-8')
    X.status(cfg, 'completed', runs=len(src_jobs)+len(tgt_jobs))
    print((X.OUT/'REPORT_ZH.md').read_text(encoding='utf-8'), flush=True)
    return 0
