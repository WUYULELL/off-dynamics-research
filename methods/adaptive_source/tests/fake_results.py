"""Fake relay results for testing analysis_tools/j014_matrix.py (local only).

usage: python fake_results.py <out_repo_dir> <real_repo_dir>
Creates results/<job>/files/code/outputs/... for the 35 j014 jobs, j014c, j015b and copies the real j011/j012
summaries so that the descriptive comparisons run."""
import json
from pathlib import Path
import random
import shutil
import sys

TASKS = ['hopper-kinematic-footjnt-medium', 'walker2d-gravity-2.0', 'walker2d-morph-leg-hard', 'hopper-kinematic-legjnt-medium',
         'walker2d-kinematic-thighjnt-hard', 'walker2d-gravity-0.5', 'hopper-morph']
SHORT = ['hopfootm', 'walkg20', 'walklegh', 'hoplegm', 'walkthih', 'walkg05', 'hopmorph']
SEEDS = [3129, 3130, 3131, 3132, 3133]
GROUPS = ['rho0', 'rho80', 'rho20T', 'rho100T', 'rho80T', 'rho80R', 'BC']
C = ['rho0', 'rho20T', 'rho80', 'rho100T', 'BC']


def out(root, job):
    d = Path(root)/'results'/job/'files'/'code'/'outputs'
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    root, real = Path(sys.argv[1]), Path(sys.argv[2])
    rng = random.Random(0)
    base = {t: {g: 30+40*rng.random() for g in GROUPS} for t in TASKS}
    last5 = {}
    for t, s in zip(TASKS, SHORT):
        for seed in SEEDS:
            job = 'j014_s%d_%s' % (seed, s)
            per = {}
            for g in GROUPS:
                v = base[t][g]+rng.gauss(0, 4)
                last5[(t, g, seed)] = v
                per[g] = dict(per_seed={str(seed): v}, mean=v, sd=None)
            (out(root, job)/'summary.json').write_text(json.dumps(dict(job=job, tasks={t: per})))
    sels, est = [], {}
    for t in TASKS:
        for seed in SEEDS:
            key = '%s|%d' % (t, seed)
            modes = {}
            for e in ['0.05', '0.1', '0.2', '0.5']:
                for m in ['main', 'abl1', 'abl2']:
                    modes['%s_eps%s' % (m, e)] = None
            ests = {}
            for g in C:
                gm = {}
                for k in modes:
                    mu = last5[(t, g, seed)]*10+rng.gauss(0, 60)
                    gm[k] = dict(mean=mu, std=abs(rng.gauss(30, 10)), J_k=[mu]*5)
                ests[g] = dict(modes=gm)
            est[key] = dict(task=t, seed=seed, estimates=ests)
            for k in modes:
                mode, _, e = k.partition('_eps')
                for kappa in [0, 0.5, 1, 1.5, 2, 3]:
                    lcb = {g: ests[g]['modes'][k]['mean']-kappa*ests[g]['modes'][k]['std'] for g in C}
                    ref = ests['rho80']['modes'][k]['mean']
                    others = [g for g in C if g != 'rho80']
                    best = max(others, key=lambda g: lcb[g])
                    chosen = best if lcb[best] > ref else 'rho80'
                    sels.append(dict(task=t, seed=seed, mode=mode, eps0=e, kappa=kappa, selected=chosen, switched=chosen != 'rho80'))
    o = out(root, 'j015b_offline_select')
    (o/'SELECTION.json').write_text(json.dumps(dict(selections=sels)))
    (o/'ESTIMATES.json').write_text(json.dumps(est))
    cal = {t: dict(IQLCAL=dict(per_seed={str(s): 40+rng.random()*10 for s in SEEDS})) for t in TASKS[:1]+TASKS[3:4]}
    (out(root, 'j014c_iqlcal')/'summary.json').write_text(json.dumps(dict(tasks=cal)))
    for job in ['j011_w50n_confirm', 'j012a_p2base_hopfootm', 'j012b_p2base_walkg20', 'j012c_p2base_walklegh', 'j012d_p2base_walkg05']:
        src = real/'results'/job/'files'/'code'/'outputs'/'summary.json'
        shutil.copyfile(str(src), str(out(root, job)/'summary.json'))
    print('fake results written')


if __name__ == '__main__':
    main()
