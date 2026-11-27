"""j015b worker: offline model-based estimate of each candidate and the LCB selection, one (task, seed).

No true return is read: inputs are the j014 policy files (verified against MODEL_MANIFEST.json, which holds
paths and hashes only), the j015a final dynamics members and H*, and the target training arrays.

Start states (N = 100, fixed per task): 50 trajectory initial states of the target training arrays used in turn,
50 target training states drawn uniformly without replacement (seeded).
Rollout of candidate c, saved policy m (Last5 points), dynamics member k: action = deterministic mean action of the
policy; next state = s + mean (s' - s) of member k, velocity columns clipped to [-10, 10]; reward = mean reward of
member k. A rollout ends (reward of that step counted) when the predicted state violates the gym v3 healthy rule
(hopper: z > 0.7, |angle| < 0.2, |obs[1:]| < 100; walker: 0.8 < z < 2.0, |angle| < 1), or before a step whose
(s, a) / state is out of the support neighbourhood (that step's reward not counted), or at the horizon.
Modes: main (H*(eps0) + fall + (s, a) distance > tau), abl1 (H* + fall), abl2 (H* + fall + state distance >
tau_state), fixed_H (H in 50/100/200/1000, fall only).
J_k = sum of rewards before the end, averaged over the N starts and then over the 5 saved policies.
Selection: LCB = mean_k J_k - kappa * std_k J_k (ddof = 1); default rho80; switch only if max LCB of the other
candidates > mean_k J_k of rho80, then to the candidate with the largest LCB.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    out = Path(job['output'])
    code = Path(job['code'])
    sys.path.insert(0, str(code))
    from asrc_common import scrub, write_json
    try:
        write_json(out/'COMPLETED.json', run(job, out, code))
    except BaseException:
        write_json(out/'FAILED.json', dict(time=time.time(), job=job['id'], traceback=scrub(traceback.format_exc())))
        print(scrub(traceback.format_exc()), flush=True)
        sys.exit(1)


def healthy(robot, s, torch):
    if robot == 'hopper':
        return (s[:, 0] > .7) & (torch.abs(s[:, 1]) < .2) & (torch.abs(s[:, 1:]) < 100.).all(1)
    return (s[:, 0] > .8) & (s[:, 0] < 2.) & (torch.abs(s[:, 1]) < 1.)


def starts_for(task_id, tar, seg, np, seed_of, n_traj_starts=50, n_uniform=50):
    """Start states (shared by j015b and j016): trajectory initial states in turn + uniform target states."""
    obs = np.asarray(tar['observations'], dtype=np.float32)
    first = [int(np.flatnonzero(seg == g)[0]) for g in np.unique(seg)]
    idx_traj = [first[i % len(first)] for i in range(n_traj_starts)]
    rng = np.random.RandomState(seed_of('starts', task_id))
    idx_uni = [int(i) for i in rng.choice(len(obs), size=n_uniform, replace=False)]
    idx = idx_traj+idx_uni
    return obs[idx], dict(rows=idx, trajectory_starts=first, n_traj_starts=n_traj_starts, n_uniform=n_uniform)


def load_policy(torch, entry, cls_otdf, cls_bc, sd, ad, device):
    from asrc_common import sha256_file
    path = Path(entry['abs_path'])
    assert sha256_file(path) == entry['sha256'], 'policy file changed: %s' % entry['path']
    blob = torch.load(str(path), map_location=device)
    if entry['kind'] == 'otdf_actor':
        net = cls_otdf(sd, ad, 1.0, 256).to(device)
        net.load_state_dict(blob)
        scale = 1.0
    else:
        net = cls_bc(sd, ad, 1., 256).to(device)
        net.load_state_dict(blob['policy'])
        scale = 1.0
    net.eval()

    def act(s):
        mu = net.network(s).chunk(2, dim=1)[0]
        return torch.tanh(mu)*scale
    return act


def run(job, out, code):
    import numpy as np
    from asrc_common import array_digest, read_json, seed_of, sha256_file, stats
    import asrc_data
    from asrc_support import Support
    if job.get('vendor'):
        sys.path.insert(0, job['vendor'])
    sys.path.insert(0, str(code/'otdf_official'))
    sys.path.insert(0, job['algo'])
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cpu'
    if job.get('device', 'cuda') == 'cuda':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(job['gpu'])
        device = 'cuda:0'
    asrc_data.import_d4rl()
    from algo.OTDF import Policy as OTDFPolicy
    from offline_offline.iql_unc import Policy as BCPolicy
    import dyn_worker

    spec = read_json(code/'tasks.json')
    task = next(t for t in spec['tasks'] if t['id'] == job['task'])
    robot = spec['robots'][task['robot']]
    sd, ad, vel = robot['state_dim'], robot['action_dim'], robot['vel_start']
    if job.get('target_override'):                       # local synthetic test only
        tp = Path(job['target_override'])
    else:
        tp = asrc_data.target_path(task, spec, [Path(p) for p in job['preflight_dirs']], job.get('private_data'))
    tar = asrc_data.call_tar_dataset_copy(tp)
    seg = asrc_data.segments_of_raw(tp)[0][:len(tar['observations'])]
    if job.get('expected_target_sas_sha256'):
        assert array_digest(asrc_data.features(tar))['sha256'] == job['expected_target_sas_sha256'], 'target array changed'
    sup = Support(tar, seg, None)
    if job.get('expected_support'):
        e = job['expected_support']
        assert abs(sup.tau-e['tau']) <= 1e-9*max(1., abs(e['tau'])) and abs(sup.tau_state-e['tau_state']) <= 1e-9*max(1., abs(e['tau_state'])), \
            'tau differs from the preflight SUPPORT.json'
    tau, tau_state = sup.tau, sup.tau_state
    robot_name = task['robot']
    T_sa = torch.as_tensor(sup.T_sa, dtype=torch.float64, device=device)
    T_s = torch.as_tensor(sup.T_s, dtype=torch.float64, device=device)
    mu = torch.as_tensor(sup.mu_sa, dtype=torch.float64, device=device)
    sdv = torch.as_tensor(sup.sd_sa, dtype=torch.float64, device=device)

    def dist(x, target):
        """Nearest standardised distance of rows x (float64) to the target set."""
        d2 = (x*x).sum(1)[:, None]+(target*target).sum(1)[None, :]-2.*x@target.T
        return torch.sqrt(torch.clamp(d2.min(1).values, min=0.))

    starts, start_info = starts_for(task['id'], tar, seg, np, seed_of)
    S0 = torch.as_tensor(starts, device=device)

    members = []
    for entry in job['dynamics']:
        assert sha256_file(entry['abs_path']) == entry['sha256'], 'dynamics member changed'
        net, blob = dyn_worker.load_member(torch, entry['abs_path'], device)
        net.eval()
        norms = [torch.as_tensor(np.asarray(blob[n], np.float32), device=device) for n in ('x_mu', 'x_sd', 'y_mu', 'y_sd')]
        members.append((net, norms))
    K = len(members)
    H = int(max(job['fixed_horizons']+[min(job['h_max'], 1000)]))
    hstar = {e: int(v) for e, v in job['h_star'].items()}
    modes = []
    for e, h in sorted(hstar.items()):
        modes += [('main', e, h, 'sa'), ('abl1', e, h, None), ('abl2', e, h, 's')]
    modes += [('fixed_%d' % h, None, h, None) for h in job['fixed_horizons']]

    def rollout(act, net, norms):
        x_mu, x_sd, y_mu, y_sd = norms
        N = S0.shape[0]
        s = S0.clone()
        rewards = torch.zeros(N, H, dtype=torch.float64, device=device)
        alive = torch.ones(N, dtype=torch.bool, device=device)
        inf = float('inf')
        end_fall = torch.full((N,), inf, dtype=torch.float64, device=device)     # rewards counted: steps < end_fall
        viol_sa = torch.full((N,), inf, dtype=torch.float64, device=device)       # first step out of (s, a) support
        viol_s = torch.full((N,), inf, dtype=torch.float64, device=device)
        nonfinite = 0
        with torch.no_grad():
            for t in range(H):
                a = act(s)
                sa = (torch.cat([s, a], 1).double()-mu)/sdv
                d_sa = dist(sa, T_sa)
                d_s = dist(sa[:, :sd], T_s)
                tt = float(t)
                viol_sa = torch.where(alive & (d_sa > tau) & torch.isinf(viol_sa), torch.full_like(viol_sa, tt), viol_sa)
                viol_s = torch.where(alive & (d_s > tau_state) & torch.isinf(viol_s), torch.full_like(viol_s, tt), viol_s)
                m_out, _ = net((torch.cat([s, a], 1)-x_mu)/x_sd)
                pred = m_out*y_sd+y_mu
                r = pred[:, sd].double()
                nxt = s+pred[:, :sd]
                nxt = torch.cat([nxt[:, :vel], torch.clamp(nxt[:, vel:], -10., 10.)], 1)
                ok = torch.isfinite(nxt).all(1) & torch.isfinite(r)
                nonfinite += int((alive & ~ok).sum())
                # non-finite prediction: the row ends before this step (reward not counted)
                end_fall = torch.where(alive & ~ok, torch.full_like(end_fall, tt), end_fall)
                live = alive & ok
                rewards[:, t] = torch.where(live, r, torch.zeros_like(r))
                h_ok = healthy(robot_name, torch.where(ok[:, None], nxt, s), torch)
                fell = live & ~h_ok
                end_fall = torch.where(fell, torch.full_like(end_fall, tt+1.), end_fall)
                alive = live & h_ok
                s = torch.where(ok[:, None], nxt, s)
                if not bool(alive.any()):
                    break
        csum = torch.cat([torch.zeros(N, 1, dtype=torch.float64, device=device), torch.cumsum(rewards, 1)], 1)
        res = {}
        for name, e, h, rule in modes:
            stop = torch.minimum(end_fall, torch.full_like(end_fall, float(h)))
            if rule == 'sa':
                stop_all = torch.minimum(stop, viol_sa)
                trunc = viol_sa < stop
            elif rule == 's':
                stop_all = torch.minimum(stop, viol_s)
                trunc = viol_s < stop
            else:
                stop_all = stop
                trunc = torch.zeros_like(alive)
            fell_first = end_fall <= torch.minimum(torch.full_like(end_fall, float(h)),
                                                   viol_sa if rule == 'sa' else (viol_s if rule == 's' else torch.full_like(end_fall, inf)))
            idx = stop_all.long().clamp(max=H)
            J = csum.gather(1, idx[:, None]).squeeze(1)
            key = name if e is None else '%s_eps%s' % (name, e)
            res[key] = dict(J=float(J.mean()), fall=float(fell_first.double().mean()), support_trunc=float(trunc.double().mean()),
                            steps=float(stop_all.mean()))
        return res, nonfinite

    estimates, deviation = {}, {}
    obs_t = torch.as_tensor(np.asarray(tar['observations'], np.float32), device=device)
    for cand in job['candidates']:
        entries = job['policies'][cand]
        assert len(entries) == 5, 'expected 5 saved policies for %s' % cand
        per_k = [[None]*len(entries) for _ in range(K)]
        devs = []
        bad = 0
        for mi, entry in enumerate(entries):
            act = load_policy(torch, entry, OTDFPolicy, BCPolicy, sd, ad, device)
            with torch.no_grad():
                a = act(obs_t)
                dd = dist((torch.cat([obs_t, a], 1).double()-mu)/sdv, T_sa)
            devs.append(dict(step=entry['step'], mean=float(dd.mean()), frac_over_tau=float((dd > tau).double().mean())))
            for k, (net, norms) in enumerate(members):
                r, nf = rollout(act, net, norms)
                bad += nf
                per_k[k][mi] = r
        # J_k: average over the 5 saved policies of the per-start mean
        modes_out = {}
        for key in per_k[0][0]:
            Jk = [sum(per_k[k][mi][key]['J'] for mi in range(len(entries)))/len(entries) for k in range(K)]
            modes_out[key] = dict(J_k=Jk, mean=sum(Jk)/K, std=stats(Jk)['sd'],
                                  fall_k=[sum(per_k[k][mi][key]['fall'] for mi in range(len(entries)))/len(entries) for k in range(K)],
                                  support_trunc_k=[sum(per_k[k][mi][key]['support_trunc'] for mi in range(len(entries)))/len(entries) for k in range(K)],
                                  steps_k=[sum(per_k[k][mi][key]['steps'] for mi in range(len(entries)))/len(entries) for k in range(K)])
        estimates[cand] = dict(modes=modes_out, nonfinite_predictions=bad)
        deviation[cand] = dict(per_policy=devs, mean=sum(d['mean'] for d in devs)/len(devs),
                               frac_over_tau=sum(d['frac_over_tau'] for d in devs)/len(devs))

    selections = []
    default = job['default']
    for key in estimates[default]['modes']:
        for kappa in job['kappas']:
            lcb = {c: estimates[c]['modes'][key]['mean']-kappa*estimates[c]['modes'][key]['std'] for c in job['candidates']}
            ref = estimates[default]['modes'][key]['mean']
            others = [c for c in job['candidates'] if c != default]
            best = max(others, key=lambda c: (lcb[c], -others.index(c)))
            chosen = best if lcb[best] > ref else default
            mode, _, eps = key.partition('_eps')
            selections.append(dict(task=job['task'], seed=job['seed'], mode=mode, eps0=eps if eps else None,
                                   horizon=(hstar[eps] if eps else int(mode.split('_')[1])), kappa=kappa,
                                   selected=chosen, switched=chosen != default, lcb=lcb, default_mean=ref))
    return dict(task=job['task'], seed=job['seed'], starts=start_info, tau=tau, tau_state=tau_state,
                h_star=hstar, members=K, estimates=estimates, deviation=deviation, selections=selections)


if __name__ == '__main__':
    main()
