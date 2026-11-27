"""j016_sim_check controller (analysis only; the selection never uses it). Runs with or after j015b.

One worker per (task, seed) runs the candidates' saved policies in the target simulator from the j015b start states
(restored from the observations, x = 0, consistency check) for H*(eps0) / 50 / 100 / 200 / 1000 steps. Then, once
j015b has completed (polled every 5 minutes up to wait_hours), it compares the model estimates with the true short
returns and relates them to Last5 (j014 summary.json, true returns, read only here) and to H* and the policy
deviation. Outputs: SIMCHECK.json, COMPARISON.json, REPORT_ZH.md, STATUS.json.
"""
import time

import asrc_common as C
import asrc_ctl as X
import asrc_queue as Q
from ctl_select import policies_for


def main(cfg):
    started = time.time()
    lc = X.local_cfg()
    X.status(cfg, 'prepare')
    X.check_common(cfg)
    gpus = Q.visible_gpus(bool(lc))
    vendor = Q.setup_mujoco(bool(lc)) or (lc['vendor'] if lc else None)
    algo = lc['algo'] if lc else Q.transfer_learner()
    spec, tasks = C.load_tasks()
    pre = X.resolve_preflight(cfg)
    dyn = X.resolve_completed(cfg['dyn_candidates'], 'dyn')
    hstar = C.read_json(dyn/'outputs'/'H_STAR.json')
    jobs = []
    for tid in cfg['tasks']:
        t = tasks[tid]
        r = spec['robots'][t['robot']]
        for seed in cfg['seeds']:
            key = '%s|%d' % (tid, seed)
            j = dict(id='sim_%s_s%d' % (t['short'], seed), worker='simcheck_worker.py', task=tid, seed=seed,
                     candidates=cfg['candidates'], fixed_horizons=cfg['fixed_horizons'], h_star=hstar['entries'][key]['h_star'],
                     policies=policies_for(cfg, tid, seed, t['short']), preflight_dirs=[str(pre)], private_data=str(C.CODE/'data'),
                     xml=str(C.CODE/t['xml']), xml_sha256=t['xml_sha256'], env_module=r['env_module'], env_class=r['env_class'],
                     vendor=vendor, algo=algo, device='cpu' if lc else 'cuda')
            if lc:
                j.update(lc.get('sim_overrides', {}).get(tid, {}))
            jobs.append(j)
    X.status(cfg, 'main', runs=len(jobs))
    stamp = time.strftime('%Y%m%d_%H%M%S')
    done, failed, _ = Q.run_queue(cfg, jobs, C.CODE/('runs_'+stamp)/'sim', gpus, 'sim', bool(lc))
    if failed:
        X.status(cfg, 'sim_failed', failed=failed)
        return 1
    sim = {}
    for j in jobs:
        c = C.read_json(done[j['id']]/'COMPLETED.json')
        sim['%s|%d' % (j['task'], j['seed'])] = c
    C.write_json(X.OUT/'SIMCHECK.json', sim)
    # ---------------- wait for j015b, then compare
    X.status(cfg, 'waiting_for_j015b')
    deadline = time.time()+3600*float(cfg['wait_hours'])
    sel_dir = None
    while sel_dir is None:
        try:
            sel_dir = X.resolve_completed(cfg['select_candidates'], 'select')
        except AssertionError:
            if time.time() > deadline:
                X.status(cfg, 'j015b_not_completed', note='true short returns saved in SIMCHECK.json; comparison not made')
                return 1
            time.sleep(1 if lc else 300)
    est = C.read_json(sel_dir/'outputs'/'ESTIMATES.json')
    dev = C.read_json(sel_dir/'outputs'/'DEVIATION.json')
    sel_hash = C.read_json(sel_dir/'outputs'/'SELECTION_SHA256.json')
    comp = compare(cfg, tasks, sim, est, dev, hstar)
    comp['selection_sha256_at_comparison'] = sel_hash
    C.write_json(X.OUT/'COMPARISON.json', comp)
    report(cfg, comp, started)
    X.status(cfg, 'completed', runs=len(jobs))
    print((X.OUT/'REPORT_ZH.md').read_text(encoding='utf-8'), flush=True)
    return 0


def last5_of(cfg, tid, seed, short, cand):
    job = cfg['j014_layout'][short][str(seed)][cand]
    s = C.read_json(X.job_dir(job)/'outputs'/'summary.json')
    return s['tasks'][tid][cand]['per_seed'][str(seed)]


def compare(cfg, tasks, sim, est, dev, hstar):
    rows, corr = [], []
    for key, c in sim.items():
        tid, seed = c['task'], c['seed']
        short = tasks[tid]['short']
        e = est[key]['estimates']
        hs = c['h_star']
        last5 = {cand: last5_of(cfg, tid, seed, short, cand) for cand in cfg['candidates']}
        for cand in cfg['candidates']:
            true = c['results'][cand]['returns']
            for mode_key, m in e[cand]['modes'].items():
                mode, _, eps = mode_key.partition('_eps')
                h = hs[eps] if eps else int(mode.split('_')[1])
                t_ret = true.get(str(h))
                rows.append(dict(task=tid, seed=seed, candidate=cand, mode=mode, eps0=eps or None, horizon=h,
                                 model_mean=m['mean'], model_std=m['std'], true_short=t_ret,
                                 error=None if t_ret is None else m['mean']-t_ret,
                                 relative_error=None if not t_ret else (m['mean']-t_ret)/abs(t_ret),
                                 deviation_mean=dev[key]['deviation'][cand]['mean'],
                                 deviation_frac_over_tau=dev[key]['deviation'][cand]['frac_over_tau'], last5=last5[cand]))
        for h in c['horizons']:
            x = [c['results'][cand]['returns'][str(h)] for cand in cfg['candidates']]
            y = [last5[cand] for cand in cfg['candidates']]
            corr.append(dict(task=tid, seed=seed, horizon=h, spearman_true_short_vs_last5=C.spearman(x, y)))
        for mode_key in e[cfg['candidates'][0]]['modes']:
            x = [e[cand]['modes'][mode_key]['mean'] for cand in cfg['candidates']]
            y = [last5[cand] for cand in cfg['candidates']]
            corr.append(dict(task=tid, seed=seed, mode=mode_key, spearman_model_vs_last5=C.spearman(x, y)))
    summary = {}
    for r in rows:
        k = '%s|%s' % (r['mode'], r['eps0'])
        s = summary.setdefault(k, dict(abs_error=[], error=[]))
        if r['error'] is not None:
            s['abs_error'].append(abs(r['error']))
            s['error'].append(r['error'])
    summary = {k: dict(mean_abs_error=C.stats(v['abs_error'])['mean'], mean_error=C.stats(v['error'])['mean'], n=len(v['error']))
               for k, v in summary.items()}
    by_h = {}
    for x in corr:
        if 'horizon' in x and x['spearman_true_short_vs_last5'] is not None:
            by_h.setdefault(x['horizon'], []).append(x['spearman_true_short_vs_last5'])
    return dict(rows=rows, correlations=corr, error_summary=summary,
                spearman_true_short_vs_last5_mean={str(h): C.stats(v)['mean'] for h, v in sorted(by_h.items())})


def report(cfg, comp, started):
    L = ['# j016_sim_check：仿真器短时真实回报核对（仅分析，选择过程不使用）', '',
         '## 模型估计相对真实短时回报的误差（模型均值 − 真实，原始回报单位）', '', '|口径|ε₀|平均误差|平均绝对误差|n|', '|---|---|---|---|---|']
    for k, v in sorted(comp['error_summary'].items()):
        mode, eps = k.split('|')
        L.append('|%s|%s|%s|%s|%d|' % (mode, eps, '%.2f' % v['mean_error'] if v['mean_error'] is not None else '',
                                       '%.2f' % v['mean_abs_error'] if v['mean_abs_error'] is not None else '', v['n']))
    L += ['', '## 短时真实回报与 Last5 的 Spearman 相关（候选间，按任务×种子平均）', '']
    for h, v in comp['spearman_true_short_vs_last5_mean'].items():
        L.append('- H = %s：%s' % (h, '%.3f' % v if v is not None else '无法计算'))
    L += ['', 'H*、偏离程度与评估误差的关系见 COMPARISON.json 的 rows（每行：任务、种子、候选、口径、H、模型估计、真实短时回报、误差、偏离程度、Last5）。',
          '总耗时 %.2f 小时。' % ((time.time()-started)/3600)]
    (X.OUT/'REPORT_ZH.md').write_text('\n'.join(L)+'\n', encoding='utf-8')
