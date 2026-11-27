"""j015b_offline_select controller (run after all j014 jobs and j015a). Reads no true return.

Inputs (server, read-only): the completed preflight job (SUPPORT.json, PREPARED_MANIFEST.json, hopper-morph target),
j015a (H_STAR.json, DYN_MANIFEST.json and the member files), every j014 job's MODEL_MANIFEST.json and policy files
(only paths and hashes; the j014 summary / eval files are never opened). One worker per (task, seed).
Outputs: SELECTION.json (selected candidate per task, seed, eps0, kappa, truncation mode) and SELECTION_SHA256.json,
ESTIMATES.json, DEVIATION.json, STATUS.json, REPORT_ZH.md.
"""
import time

import asrc_common as C
import asrc_ctl as X
import asrc_queue as Q


def policies_for(cfg, tid, seed, short):
    """Saved policies of the candidates of (task, seed) from the j014 job(s) holding them."""
    out = {}
    for cand in cfg['candidates']:
        job = cfg['j014_layout'][short][str(seed)][cand]
        d = X.job_dir(job)
        st = C.read_json(d/'outputs'/'STATUS.json')
        assert st.get('phase') == 'completed', 'j014 job %s not completed' % job
        man = C.read_json(d/'outputs'/'MODEL_MANIFEST.json')
        e = man['entries']['%s|%s|%d' % (tid, cand, seed)]
        out[cand] = [dict(m, abs_path=str(d/m['path']), job=job) for m in sorted(e['models'], key=lambda x: x['step'])]
    return out


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
    cost_m, support_m, prepared_m = X.preflight_files(pre)
    dyn = X.resolve_completed(cfg['dyn_candidates'], 'dyn')
    hstar = C.read_json(dyn/'outputs'/'H_STAR.json')
    dman = C.read_json(dyn/'outputs'/'DYN_MANIFEST.json')
    jobs = []
    for tid in cfg['tasks']:
        t = tasks[tid]
        for seed in cfg['seeds']:
            key = '%s|%d' % (tid, seed)
            members = [dict(m, abs_path=str(dyn/m['path'])) for m in dman['entries'][key]['members']]
            j = dict(id='select_%s_s%d' % (t['short'], seed), worker='select_worker.py', task=tid, seed=seed,
                     candidates=cfg['candidates'], default=cfg['default'], kappas=cfg['kappas'],
                     fixed_horizons=cfg['fixed_horizons'], h_star=hstar['entries'][key]['h_star'], h_max=hstar['entries'][key]['h_max'],
                     dynamics=members, policies=policies_for(cfg, tid, seed, t['short']), preflight_dirs=[str(pre)],
                     private_data=str(C.CODE/'data'), expected_target_sas_sha256=prepared_m['target'][tid]['digest']['features_sas']['sha256'],
                     expected_support=dict(tau=support_m['tasks'][tid]['tau'], tau_state=support_m['tasks'][tid]['tau_state']),
                     vendor=vendor, algo=algo, device='cpu' if lc else 'cuda')
            if lc:
                j.update(lc.get('select_overrides', {}).get(tid, {}))
            jobs.append(j)
    X.status(cfg, 'main', runs=len(jobs))
    stamp = time.strftime('%Y%m%d_%H%M%S')
    done, failed, _ = Q.run_queue(cfg, jobs, C.CODE/('runs_'+stamp)/'select', gpus, 'select', bool(lc))
    if failed:
        X.status(cfg, 'select_failed', failed=failed)
        return 1
    selections, estimates, deviation = [], {}, {}
    for j in jobs:
        c = C.read_json(done[j['id']]/'COMPLETED.json')
        key = '%s|%d' % (j['task'], j['seed'])
        selections += c['selections']
        estimates[key] = dict(task=j['task'], seed=j['seed'], h_star=c['h_star'], tau=c['tau'], tau_state=c['tau_state'],
                              starts=c['starts'], estimates=c['estimates'])
        deviation[key] = dict(task=j['task'], seed=j['seed'], deviation=c['deviation'])
    sel = dict(job=cfg['job'], candidates=cfg['candidates'], default=cfg['default'], kappas=cfg['kappas'],
               rule='LCB = mean_k J_k - kappa * std_k J_k (ddof 1); default rho80; switch only when the largest LCB of the '
                    'other candidates exceeds mean_k J_k of rho80, then to that candidate', selections=selections)
    C.write_json(X.OUT/'SELECTION.json', sel)
    C.write_json(X.OUT/'SELECTION_SHA256.json', dict(file='SELECTION.json', sha256=C.sha256_file(X.OUT/'SELECTION.json'),
                                                    written=C.utc()))
    C.write_json(X.OUT/'ESTIMATES.json', estimates)
    C.write_json(X.OUT/'DEVIATION.json', deviation)
    L = ['# j015b_offline_select：基于动力学模型的离线选择（不读任何真实回报）', '',
         '候选 %s，默认 %s；κ ∈ %s；截断口径 main / abl1 / abl2（每档 ε₀）与固定步长 %s。' % (
             cfg['candidates'], cfg['default'], cfg['kappas'], cfg['fixed_horizons']), '',
         '## 主口径各 (ε₀, κ) 的改选次数（%d 个任务×种子）' % len(jobs), '', '|ε₀|' + '|'.join('κ=%g' % k for k in cfg['kappas']) + '|',
         '|---|' + '---|'*len(cfg['kappas'])]
    eps_list = sorted({s['eps0'] for s in selections if s['mode'] == 'main'}, key=float)
    for e in eps_list:
        L.append('|%s|%s|' % (e, '|'.join(str(sum(1 for s in selections if s['mode'] == 'main' and s['eps0'] == e and s['kappa'] == k and s['switched']))
                                         for k in cfg['kappas'])))
    L += ['', 'SELECTION.json 的 SHA-256 记在 SELECTION_SHA256.json。总耗时 %.2f 小时。' % ((time.time()-started)/3600)]
    (X.OUT/'REPORT_ZH.md').write_text('\n'.join(L)+'\n', encoding='utf-8')
    X.status(cfg, 'completed', runs=len(jobs), selection_sha256=C.sha256_file(X.OUT/'SELECTION.json'))
    print((X.OUT/'REPORT_ZH.md').read_text(encoding='utf-8'), flush=True)
    return 0
