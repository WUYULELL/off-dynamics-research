"""CPU diagnosis and static selection; selection audit precedes mask sealing."""
from runtime import *
import argparse
import numpy as np
import torch
from scipy.spatial import cKDTree
from learner import initialize

def diagnosis():
    check_frozen()
    selected = read(OUT / 'T_selected.json')
    ref = read(OUT / 'reference_ready.json')
    assert sha(ref['model']) == ref['model_sha256']
    out = OUT / 'diagnosis'
    fresh(out)
    configure_cpu()
    model = initialize(model_config(selected['temp'], selected['steps'], 1001), 'cpu')
    model.restore(ref['model'])
    e = env()
    fields = {k: [] for k in ('state','action','next_state','reward','not_done','episode','env_seed','truncated')}
    episode, episode_step = 0, 0
    e.seed(200000)
    s = e.reset()
    for i in range(5000):
        a = model.deterministic_action(s)
        ns, r, done, info = e.step(a)
        truncated = bool(info.get('TimeLimit.truncated', False))
        values = (s.copy(), a.copy(), ns.copy(), [r], [float(not (done and not truncated))],
                  episode, 200000 + episode, truncated)
        for k, v in zip(fields, values):
            fields[k].append(v)
        s = ns
        episode_step += 1
        if done and i < 4999:
            episode += 1
            episode_step = 0
            e.seed(200000 + episode)
            s = e.reset()
    e.close()
    for k in fields:
        fields[k] = np.asarray(fields[k], dtype=np.float32 if k in ('state','action','next_state','reward','not_done') else None)
    assert all(len(v) == 5000 for v in fields.values())
    np.savez_compressed(out / 'trajectories.npz', **fields)
    write(out / 'completed.json', dict(steps=5000, episodes_started=episode + 1,
        reference_model_sha256=ref['model_sha256'], diagnostic_env_seeds=list(range(200000, 200001 + episode)),
        budget_cut_does_not_change_not_done=True, last_episode_steps=episode_step,
        final_record_is_budget_cut=not bool(done), policy='fixed T mean',
        reference_distribution_only=True, noniterative=True,
        counts_exclude_evaluation_and_old_source_cache_queries=True,
        arrays_sha256=sha(out / 'trajectories.npz')))

def rms_distance(tree, x, k=20, leave_self=False):
    # scipy 1.4.1 spells the worker argument n_jobs, not workers.
    d, _ = tree.query(x, k=k + int(leave_self), n_jobs=2)
    if leave_self:
        d = d[:, 1:]
    return np.sqrt(np.mean(d ** 2, axis=1) / x.shape[1])

def component_audit(logrel, loggap, order, ratios):
    n = len(order)
    reports = []
    for idx, p in enumerate(ratios + [1.]):
        # At p=1 this additionally reports the global top-5%, which is common to all p>=5%.
        subset_n = int(np.floor(p * n))
        count = max(2, int(np.floor(.05 * subset_n)))
        ids = order[:count]
        lr, lg = logrel[ids], loggap[ids]
        rng = np.random.RandomState(seed('score_contribution_%d' % idx))
        i = rng.randint(count, size=100000)
        j = rng.randint(count, size=100000)
        a, b = lr[i] - lr[j], lg[i] - lg[j]
        delta = a + b
        abs_a, abs_b = float(np.abs(a).mean()), float(np.abs(b).mean())
        total = abs_a + abs_b
        variation = dict(rel=abs_a / total if total else None, gap=abs_b / total if total else None)
        valid = np.abs(delta) > 1e-12
        if valid.any():
            vr = float((np.sign(a[valid]) * np.sign(delta[valid])).mean())
            vg = float((np.sign(b[valid]) * np.sign(delta[valid])).mean())
            # Two-feature Shapley allocation of agreement with the full ranking.
            rank = dict(rel=float(.5 * (1 + vr - vg)), gap=float(.5 * (1 + vg - vr)))
        else:
            rank = dict(rel=None, gap=None)
        dominance = [key for key in ('rel','gap') if any(d[key] is not None and d[key] > .9 for d in (variation, rank))]
        # Deletion check over the full source pool, not just within selected rows.
        overlap = {}
        for key, comp in (('rel', logrel), ('gap', loggap)):
            alone = np.lexsort((np.arange(n), -comp))[:count]
            overlap[key] = float(len(np.intersect1d(ids, alone)) / count)
        summaries = {k: dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()),
            sd=float(v.std()), distinct=int(len(np.unique(v)))) for k, v in (('logrel', lr), ('loggap', lg))}
        reports.append(dict(retention_ratio=p, selected_n=subset_n, audited_n=count,
            full_source_fraction=count / n, pair_count=100000, non_tied_pair_fraction=float(valid.mean()),
            absolute_pair_margin_contribution=variation, ranking_agreement_shapley_contribution=rank,
            delete_other_component_top_set_overlap=overlap, components=summaries,
            dominant_components_over_90pct=dominance,
            interpretation='本次G实质为单一分量选择；不得宣称组合效果' if dominance else
                '未触发90%主导；这不是组合效果的因果证据'))
    return dict(passed=True, zero_additional_training=True,
        top5_definition='each retained p*N subset top5%; plus common full-source top5% at p=1',
        contribution_definition='mean absolute pair log-score difference share; also two-feature Shapley allocation of signed pair rank agreement',
        dominance_rule='rel or gap exceeds .90 in either variation or rank contribution; conservative reporting only',
        ties='total differences <=1e-12 omitted from rank metric; both-constant contributions null',
        no_score_retuning=True, fixed_k=20, fixed_relevance_weights=[.5,.5], results=reports)

def score():
    check_frozen()
    configure_cpu()
    out = OUT / 'scores'
    fresh(out)
    assert read(OUT / 'diagnosis/completed.json')['steps'] == 5000
    target = target_data()
    raw, oracle, raw_ids, meta = source_data()
    with np.load(OUT / 'diagnosis/trajectories.npz') as f:
        visits = f['state'].astype(np.float64)
    mu = target[0].astype(np.float64).mean(axis=0)
    sd = np.maximum(target[0].astype(np.float64).std(axis=0, ddof=0), 1e-3)
    banks = [target[0], target[2][target[4][:,0] == 1], visits]
    trees, bandwidths = [], []
    for bank in banks:
        z = (bank.astype(np.float64) - mu) / sd
        tree = cKDTree(z)
        h = max(float(np.median(rms_distance(tree, z, leave_self=True))), 1e-3)
        trees.append(tree)
        bandwidths.append(h)
    n = len(raw_ids)
    logrel, loggap, cs = np.empty(n), np.empty(n), np.empty(n)
    selected, ref = read(OUT / 'T_selected.json'), read(OUT / 'reference_ready.json')
    assert sha(ref['model']) == ref['model_sha256']
    model = initialize(model_config(selected['temp'], selected['steps'], 1001), 'cpu')
    model.restore(ref['model'])
    for start in range(0, n, 4096):
        end = min(start + 4096, n)
        z = (raw[0][start:end].astype(np.float64) - mu) / sd
        d_t, d_n, d_pi = [rms_distance(t, z) for t in trees]
        rel = np.logaddexp(-.5*(d_pi/bandwidths[2])**2,
                          -.5*(d_n/bandwidths[1])**2) - np.log(2.)
        gap = np.log(np.maximum(-np.expm1(-.5*(d_t/bandwidths[0])**2), 1e-12))
        logrel[start:end], loggap[start:end] = rel, gap
        with torch.no_grad():
            vraw = model.v_func(torch.as_tensor(raw[2][start:end], dtype=torch.float32)).numpy().reshape(-1)
            voracle = model.v_func(torch.as_tensor(np.array(oracle[2][start:end], copy=True), dtype=torch.float32)).numpy().reshape(-1)
        cs[start:end] = np.abs(raw[3][start:end,0] + .99*raw[4][start:end,0]*vraw -
            oracle[3][start:end,0] - .99*oracle[4][start:end,0]*voracle)
        write(out / 'progress.json', dict(rows=end, total=n, pid=os.getpid()), replace=True)
    assert all(np.isfinite(a).all() for a in (logrel, loggap, cs))
    order = np.lexsort((np.arange(n), -(logrel + loggap)))
    corder = np.lexsort((np.arange(n), cs))
    ratios = [.05, .2, .5]
    audit = component_audit(logrel, loggap, order, ratios)
    # Persist the audit BEFORE sealing masks. Audit never changes the score rule.
    write(out / 'component_audit.json', audit)
    lines = ['# G零训练分数分布审计', '',
        '前5%按每个比例选中子集内部计算；另列全池前5%，不能将重叠审计视为独立证据。', '',
        '贡献同时报告成对log分差幅度份额和成对排名一致性的双分量Shapley分配；任一超过90%就按单分量描述。', '',
        '|保留比例|审计行数|rel幅度份额|gap幅度份额|rel排名贡献|gap排名贡献|解释|',
        '|---|---:|---:|---:|---:|---:|---|']
    for x in audit['results']:
        vals = [x[k][v] for k in ('absolute_pair_margin_contribution','ranking_agreement_shapley_contribution') for v in ('rel','gap')]
        lines.append('|%g%%|%d|%s|%s|%s|%s|%s|' % tuple([100*x['retention_ratio'],x['audited_n']] +
            ['不识别' if v is None else '%.2f%%' % (100*v) for v in vals] + [x['interpretation']]))
    (out / 'SCORE_AUDIT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    np.savez_compressed(out / 'components.npz', logrel=logrel, loggap=loggap, C=cs,
        target_mean=mu, target_std=sd, bandwidths=bandwidths)
    masks = {}
    for p in ratios:
        count = int(np.floor(p*n))
        for arm, ranking in (('G',order),('C',corder)):
            file = out / ('%s_%02d.npy' % (arm, round(100*p)))
            np.save(file, np.sort(ranking[:count]))
            masks[file.name] = dict(n=count, sha256=sha(file))
    write(out / 'completed.json', dict(passed=True, n=n, denominator='full canonical source rows',
        masks=masks, bandwidths=bandwidths, target_next_bank_rows=len(banks[1]),
        reference_model_sha256=ref['model_sha256'], audit_sha256=sha(out/'component_audit.json'),
        source_row_ids_sha256=meta['raw_indices_sha256'],
        C_limitation='V0仅由5000 target学得，缺口外推不可靠；G胜C仅表示几何缺口信号更可靠，不是原则优劣',
        task2_transfer='percentage, never absolute row count'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--action', required=True, choices=('diagnosis','score'))
    a = p.parse_args()
    (diagnosis if a.action == 'diagnosis' else score)()
