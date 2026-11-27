"""Analysis of the adaptive-source batch (j014 candidate matrix, j015b selection, j016 simulator check).

Reads the files the relay retrieved (results/<job>/files/code/outputs/...), applies the rules registered in
research/PLAN.md (j013-j016 entry, written before any result) and writes analysis/j014_matrix.md and
analysis/j014_matrix.json. Python 3 standard library only.

usage: python analysis_tools/j014_matrix.py [--repo .] [--layout single|split] [--out analysis/j014_matrix]
"""
import argparse
import json
import math
from pathlib import Path

TASKS = ['hopper-kinematic-footjnt-medium', 'walker2d-gravity-2.0', 'walker2d-morph-leg-hard', 'hopper-kinematic-legjnt-medium',
         'walker2d-kinematic-thighjnt-hard', 'walker2d-gravity-0.5', 'hopper-morph']
SCREEN = TASKS[:6]
SHORT = {'hopper-kinematic-footjnt-medium': 'hopfootm', 'walker2d-gravity-2.0': 'walkg20', 'walker2d-morph-leg-hard': 'walklegh',
         'hopper-kinematic-legjnt-medium': 'hoplegm', 'walker2d-kinematic-thighjnt-hard': 'walkthih', 'walker2d-gravity-0.5': 'walkg05',
         'hopper-morph': 'hopmorph'}
SEEDS = [3129, 3130, 3131, 3132, 3133]
C = ['rho0', 'rho20T', 'rho80', 'rho100T', 'BC']
DEFAULT = 'rho80'
EPS = ['0.05', '0.1', '0.2', '0.5']
KAPPAS = [0, 0.5, 1, 1.5, 2, 3]
OLD = {'hopper-kinematic-footjnt-medium': 'j012a_p2base_hopfootm', 'walker2d-gravity-2.0': 'j012b_p2base_walkg20',
       'walker2d-morph-leg-hard': 'j012c_p2base_walklegh', 'hopper-kinematic-legjnt-medium': 'j011_w50n_confirm',
       'walker2d-kinematic-thighjnt-hard': 'j011_w50n_confirm', 'walker2d-gravity-0.5': 'j012d_p2base_walkg05'}
SPLIT = {'A': ['rho80', 'rho0', 'rho20T', 'BC'], 'B': ['rho80', 'rho100T', 'rho80T', 'rho80R']}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v)/len(v) if v else None


def sd(v):
    v = [x for x in v if x is not None]
    if len(v) < 2:
        return None
    m = sum(v)/len(v)
    return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))


def spearman(x, y):
    n = len(x)
    if n < 2:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.]*n
        i = 0
        while i < n:
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]:
                j += 1
            for k in range(i, j+1):
                r[order[k]] = (i+j)/2.+1.
            i = j+1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx)/n, sum(ry)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    sxx, syy = sum((a-mx)**2 for a in rx), sum((b-my)**2 for b in ry)
    return None if sxx <= 0 or syy <= 0 else sxy/math.sqrt(sxx*syy)


def outputs(repo, job):
    return Path(repo)/'results'/job/'files'/'code'/'outputs'


def job_for(layout, task, seed, group):
    if layout == 'single':
        return 'j014_s%d_%s' % (seed, SHORT[task])
    part = 'A' if group in SPLIT['A'] else 'B'
    return 'j014_s%d_%s_%s' % (seed, SHORT[task], part)


def load_matrix(repo, layout):
    """last5[task][group][seed] from the j014 summaries; missing jobs listed."""
    last5, missing = {}, []
    for t in TASKS:
        last5[t] = {}
        for g in C+['rho80T', 'rho80R']:
            last5[t][g] = {}
            for s in SEEDS:
                f = outputs(repo, job_for(layout, t, s, g))/'summary.json'
                if not f.is_file():
                    missing.append(job_for(layout, t, s, g))
                    continue
                v = read(f)['tasks'][t][g]['per_seed'][str(s)]
                last5[t][g][s] = v
    return last5, sorted(set(missing))


def gate(last5):
    rows, gaps7, gaps6 = {}, [], []
    for t in TASKS:
        means = {g: mean([last5[t][g].get(s) for s in SEEDS]) for g in C}
        if any(v is None for v in means.values()):
            rows[t] = dict(complete=False)
            continue
        o = max(C, key=lambda g: means[g])
        gap = means[o]-means[DEFAULT]
        per_seed_max = mean([max(last5[t][g][s] for g in C)-last5[t][DEFAULT][s] for s in SEEDS])
        rows[t] = dict(complete=True, means=means, O_t=means[o], O_group=o, D_t=means[DEFAULT], gap=gap,
                       per_seed_max_minus_rho80=per_seed_max)
        gaps7.append(gap)
        if t in SCREEN:
            gaps6.append(gap)
    out = dict(rows=rows)
    if len(gaps7) == len(TASKS):
        H = mean(gaps7)
        n5 = sum(1 for t in TASKS if rows[t]['gap'] >= 5)
        if H >= 3 and n5 >= 2:
            decision = 'continue innovation 1 (H >= 3 and >= 2 tasks with O_t - D_t >= 5)'
        elif H < 1.5 or n5 == 0:
            decision = 'stop innovation 1 (H < 1.5 or no task with O_t - D_t >= 5)'
        else:
            decision = 'other: only shrink the candidate set, no new candidate levels'
        out.update(H7=H, tasks_gap_ge_5=n5, decision=decision, H6=mean(gaps6),
                   tasks_gap_ge_5_screen=sum(1 for t in SCREEN if rows[t]['gap'] >= 5))
    else:
        out['decision'] = 'not computed: incomplete matrix'
    return out


def selections(repo, sel_job='j015b_offline_select'):
    f = outputs(repo, sel_job)/'SELECTION.json'
    e = outputs(repo, sel_job)/'ESTIMATES.json'
    if not f.is_file() or not e.is_file():
        return None, None
    return read(f), read(e)


def choose_eps_kappa(last5, sel, est):
    """Registered rule on the 6 screening tasks: eps0 = highest mean Spearman (model mean J vs true Last5 across the
    candidates, per task x seed); kappa = smallest with wrong-switch rate <= 20% (30 decisions pooled), else 3."""
    eps_scores = {}
    for e in EPS:
        vals = []
        for t in SCREEN:
            for s in SEEDS:
                key = '%s|%d' % (t, s)
                if key not in est:
                    continue
                x = [est[key]['estimates'][g]['modes']['main_eps%s' % e]['mean'] for g in C]
                y = [last5[t][g].get(s) for g in C]
                if None in y:
                    continue
                r = spearman(x, y)
                if r is not None:
                    vals.append(r)
        eps_scores[e] = dict(mean_spearman=mean(vals), n=len(vals))
    valid = [e for e in EPS if eps_scores[e]['mean_spearman'] is not None]
    if not valid:
        return None, None, eps_scores, {}
    best = max(valid, key=lambda e: (eps_scores[e]['mean_spearman'], -float(e)))      # ties -> smaller eps0
    rates = {}
    for k in KAPPAS:
        wrong, n, switches = 0, 0, 0
        for x in sel['selections']:
            if x['mode'] != 'main' or x['eps0'] != best or x['kappa'] != k or x['task'] not in SCREEN:
                continue
            t, s = x['task'], x['seed']
            n += 1
            if x['switched']:
                switches += 1
                if last5[t][x['selected']][s] < last5[t][DEFAULT][s]:
                    wrong += 1
        rates[str(k)] = dict(decisions=n, switches=switches, wrong_switches=wrong,
                             wrong_switch_rate=wrong/n if n else None, wrong_among_switches=wrong/switches if switches else None)
    ok = [k for k in KAPPAS if rates[str(k)]['wrong_switch_rate'] is not None and rates[str(k)]['wrong_switch_rate'] <= 0.2]
    kappa = min(ok) if ok else 3
    return best, kappa, eps_scores, rates


def screening(cand_means, otdf_means, bc_means):
    """PLAN section 1 phase-2 screening: (1) 6-task mean > OTDF, (2) >= 4 tasks >= OTDF, (3) 6-task mean >= BC."""
    m_c, m_o, m_b = mean([cand_means[t] for t in SCREEN]), mean([otdf_means[t] for t in SCREEN]), mean([bc_means[t] for t in SCREEN])
    n = sum(1 for t in SCREEN if cand_means[t] >= otdf_means[t])
    return dict(mean6=m_c, otdf_mean6=m_o, bc_mean6=m_b, criterion1=m_c > m_o, tasks_at_least_otdf=n, criterion2=n >= 4,
                criterion3=m_c >= m_b, effective=bool(m_c > m_o and n >= 4 and m_c >= m_b))


def selection_quality(last5, sel):
    out = {}
    for x in sel['selections']:
        t, s = x['task'], x['seed']
        if None in [last5[t][g].get(s) for g in C]:
            continue
        key = '%s|%s|%s' % (x['mode'], x['eps0'], x['kappa'])
        q = out.setdefault(key, dict(n=0, correct=0, regret=[], gain=[], switches=0, wrong=0))
        best = max(last5[t][g][s] for g in C)
        v = last5[t][x['selected']][s]
        q['n'] += 1
        q['correct'] += int(v == best)
        q['regret'].append(best-v)
        q['gain'].append(v-last5[t][DEFAULT][s])
        if x['switched']:
            q['switches'] += 1
            q['wrong'] += int(v < last5[t][DEFAULT][s])
    return {k: dict(n=q['n'], accuracy=q['correct']/q['n'], regret_mean=mean(q['regret']), gain_vs_rho80_mean=mean(q['gain']),
                    switch_rate=q['switches']/q['n'], wrong_switch_rate=q['wrong']/q['n']) for k, q in out.items() if q['n']}


def old_groups(repo, task, group):
    job = OLD.get(task)
    if not job:
        return None
    f = outputs(repo, job)/'summary.json'
    if not f.is_file():
        return None
    g = read(f)['tasks'][task]['groups'].get(group)
    return None if g is None else {int(k): v for k, v in g['per_seed'].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--layout', default='single', choices=['single', 'split'])
    ap.add_argument('--out', default='analysis/j014_matrix')
    a = ap.parse_args()
    repo = Path(a.repo)
    last5, missing = load_matrix(repo, a.layout)
    res = dict(missing_jobs=missing, gate=gate(last5))
    means = {t: {g: mean([last5[t][g].get(s) for s in SEEDS]) for g in C+['rho80T', 'rho80R']} for t in TASKS}
    res['means'] = means
    sel, est = selections(repo)
    if sel is not None and not missing:
        eps, kappa, eps_scores, rates = choose_eps_kappa(last5, sel, est)
        res['eps0_selection'] = eps_scores
        res['kappa_selection'] = rates
        res['eps0'], res['kappa'] = eps, kappa
        res['selection_quality'] = selection_quality(last5, sel)
        if eps is not None:
            sel_means = {}
            for t in TASKS:
                v = []
                for x in sel['selections']:
                    if x['task'] == t and x['mode'] == 'main' and x['eps0'] == eps and x['kappa'] == kappa:
                        v.append(last5[t][x['selected']][x['seed']])
                sel_means[t] = mean(v)
            res['SEL_means'] = sel_means
            res['SEL_screening'] = screening(sel_means, {t: means[t]['rho80'] for t in TASKS}, {t: means[t]['BC'] for t in TASKS})
            res['SEL_mean7'] = mean(list(sel_means.values()))
            by_mode = {}
            for mode in ('main', 'abl1', 'abl2'):
                by_mode[mode] = {t: [x['selected'] for x in sel['selections'] if x['task'] == t and x['mode'] == mode
                                     and x['eps0'] == eps and x['kappa'] == kappa] for t in TASKS}
            res['selections_by_truncation_mode'] = by_mode
    if not missing:
        r80t = {t: means[t]['rho80T'] for t in TASKS}
        sc = screening(r80t, {t: means[t]['rho80'] for t in TASKS}, {t: means[t]['BC'] for t in TASKS})
        sc['above_rho80R_mean6'] = mean([r80t[t] for t in SCREEN]) > mean([means[t]['rho80R'] for t in SCREEN])
        sc['effective_with_rho80R'] = bool(sc['effective'] and sc['above_rho80R_mean6'])
        res['rho80T_screening'] = sc
    desc = {}
    for t in TASKS:
        bc_old, t_old = old_groups(repo, t, 'BC'), old_groups(repo, t, 'T')
        d = {}
        if bc_old and all(s in bc_old and last5[t]['BC'].get(s) is not None for s in SEEDS):
            diff = [last5[t]['BC'][s]-bc_old[s] for s in SEEDS]
            d['BC_retrained_minus_j011_j012'] = dict(mean=mean(diff), per_seed=diff)
        if t_old and all(s in t_old and last5[t]['rho0'].get(s) is not None for s in SEEDS):
            diff = [last5[t]['rho0'][s]-t_old[s] for s in SEEDS]
            d['rho0_minus_T'] = dict(mean=mean(diff), per_seed=diff)
        desc[t] = d
    cal = outputs(repo, 'j014c_iqlcal')/'summary.json'
    if cal.is_file():
        c = read(cal)
        for t, per in c['tasks'].items():
            t_old = old_groups(repo, t, 'T')
            v = per['IQLCAL']['per_seed']
            d = desc.setdefault(t, {})
            d['j014c_IQL'] = dict(mean=mean(list(v.values())), sd=sd(list(v.values())))
            if t_old:
                d['j014c_minus_T'] = mean([v[str(s)]-t_old[s] for s in SEEDS])
    res['descriptive'] = desc
    pre = [p for p in sorted((repo/'results').glob('j013*_adapt_preflight*')) if (p/'files/code/outputs/PREFLIGHT.json').is_file()]
    if pre:
        res['near_layer_fractions'] = read(pre[-1]/'files/code/outputs/PREFLIGHT.json').get('support')
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(res, out.with_suffix('.md'))
    print(out.with_suffix('.md').read_text(encoding='utf-8'))


def f2(x):
    return '' if x is None else '%.2f' % x


def write_report(res, path):
    L = ['# j014 候选回报矩阵、离线选择与仿真核对（按 research/PLAN.md 登记的规则计算）', '']
    if res['missing_jobs']:
        L += ['缺少 %d 个 j014 任务的结果：%s' % (len(res['missing_jobs']), ', '.join(res['missing_jobs'][:10])), '']
    g = res['gate']
    L += ['## 门槛（候选集合 C = ρ0, ρ20T, ρ80, ρ100T, BC；5 个种子 Last5 均值）', '',
          '|任务|ρ0|ρ20T|ρ80|ρ100T|BC|O_t（组）|O_t − D_t|按种子取最大 − ρ80（带噪声上界）|', '|---|---|---|---|---|---|---|---|---|']
    for t in TASKS:
        r = g['rows'][t]
        if not r.get('complete'):
            L.append('|%s|（不全）||||||||' % t)
            continue
        m = r['means']
        L.append('|%s|%s|%s|%s|%s|%s|%s（%s）|%s|%s|' % (t, f2(m['rho0']), f2(m['rho20T']), f2(m['rho80']), f2(m['rho100T']), f2(m['BC']),
                                                       f2(r['O_t']), r['O_group'], f2(r['gap']), f2(r['per_seed_max_minus_rho80'])))
    L += ['', 'H（7 个任务）= %s；O_t − D_t ≥ 5 的任务数 %s；6 个筛查任务版本 H = %s（%s 个任务 ≥ 5）。' % (
        f2(g.get('H7')), g.get('tasks_gap_ge_5'), f2(g.get('H6')), g.get('tasks_gap_ge_5_screen')),
          '**判定：%s**' % g['decision'], '']
    if 'eps0' in res:
        L += ['## κ 与 ε₀ 的确定（6 个筛查任务；同一批数据上调参，结论偏乐观，须用新种子 3134 及以后确认）', '']
        for e, v in res['eps0_selection'].items():
            L.append('- ε₀ = %s：模型均值 Ĵ 与真实 Last5 的候选间 Spearman 平均 %s（n = %d）' % (e, f2(v['mean_spearman']), v['n']))
        for k, v in res['kappa_selection'].items():
            L.append('- κ = %s：改选 %d 次，错误切换 %d 次，错误切换率 %s（错误/改选 %s）' % (
                k, v['switches'], v['wrong_switches'], f2(v['wrong_switch_rate']), f2(v['wrong_among_switches'])))
        L += ['- 确定：ε₀ = %s，κ = %s' % (res['eps0'], res['kappa']), '']
    if res.get('selection_quality') and res.get('eps0') is not None:
        q = res['selection_quality']
        L += ['## 选择的正确率与遗憾（7 个任务 × 5 个种子；ε₀ = %s；固定步长口径不依赖 ε₀）' % res['eps0'], '',
              '|口径|κ|正确率|平均遗憾|相对 ρ80 的平均增益|改选率|错误切换率|', '|---|---|---|---|---|---|---|']
        for mode in ('main', 'abl1', 'abl2'):
            for k in KAPPAS:
                v = q.get('%s|%s|%s' % (mode, res['eps0'], k))
                if v:
                    L.append('|%s|%s|%.2f|%s|%s|%.2f|%.2f|' % (mode, k, v['accuracy'], f2(v['regret_mean']), f2(v['gain_vs_rho80_mean']),
                                                              v['switch_rate'], v['wrong_switch_rate']))
        for h in (50, 100, 200, 1000):
            v = q.get('fixed_%d|None|%s' % (h, res['kappa']))
            if v:
                L.append('|fixed_%d|%s|%.2f|%s|%s|%.2f|%.2f|' % (h, res['kappa'], v['accuracy'], f2(v['regret_mean']),
                                                                f2(v['gain_vs_rho80_mean']), v['switch_rate'], v['wrong_switch_rate']))
        L.append('')
    if 'SEL_screening' in res:
        s = res['SEL_screening']
        L += ['## SEL（主截断方式）对照 OTDF（PLAN 第一节第二阶段三条筛查标准）', '',
              '6 任务平均：SEL %s，OTDF(ρ80) %s，BC %s；≥ OTDF 的任务数 %d；标准 1/2/3：%s/%s/%s → %s' % (
                  f2(s['mean6']), f2(s['otdf_mean6']), f2(s['bc_mean6']), s['tasks_at_least_otdf'], s['criterion1'], s['criterion2'],
                  s['criterion3'], '筛查有效' if s['effective'] else '筛查无效'), '']
    if 'rho80T_screening' in res:
        s = res['rho80T_screening']
        L += ['## ρ80T 对照 OTDF（三条标准）并要求 6 任务平均高于 ρ80R', '',
              '6 任务平均：ρ80T %s，OTDF %s，BC %s；≥ OTDF 的任务数 %d；高于 ρ80R：%s → %s' % (
                  f2(s['mean6']), f2(s['otdf_mean6']), f2(s['bc_mean6']), s['tasks_at_least_otdf'], s['above_rho80R_mean6'],
                  '筛查有效' if s['effective_with_rho80R'] else '筛查无效'), '']
    L += ['## 描述性内容（不判定）', '']
    for t, d in res['descriptive'].items():
        if d:
            L.append('- %s：%s' % (t, json.dumps({k: (v['mean'] if isinstance(v, dict) and 'mean' in v else v) for k, v in d.items()}, ensure_ascii=False)))
    if res.get('selections_by_truncation_mode'):
        L.append('- 三种截断方式的选择结果：见 JSON 的 selections_by_truncation_mode')
    if res.get('near_layer_fractions'):
        L.append('- 近邻层比例：见 JSON 的 near_layer_fractions（来自 j013 PREFLIGHT.json）')
    Path(path).write_text('\n'.join(L)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
