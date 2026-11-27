"""j016 worker: true short-horizon returns in the target simulator from the j015b start states, one (task, seed).

Analysis only; the selection never uses this output. For every start state the simulator state is restored from
the observation (qpos = [0, obs[:nq-1]] with the x position set to 0, qvel = obs[nq-1:]) after env.reset(), and
the restored observation is checked against the start observation (max |difference| recorded; > 1e-6 fails).
Each saved policy (5 Last5 points of each candidate) then acts with its deterministic mean action for up to 1000
steps (gym v3 termination, TimeLimit 1000); returns are summed up to H in {H*(eps0) ..., 50, 100, 200, 1000}.
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


def make_envs(job, n, sd, ad):
    if job.get('env_factory'):
        sys.path.insert(0, str(Path(job['env_factory']).parent))
        mod = __import__(Path(job['env_factory']).stem)
        return [mod.make(sd, ad) for _ in range(n)]
    import gym
    constructor = getattr(__import__(job['env_module'], fromlist=[job['env_class']]), job['env_class'])
    envs = [gym.wrappers.TimeLimit(constructor(xml_file=job['xml']), max_episode_steps=1000) for _ in range(n)]
    for e in envs:
        assert e.observation_space.shape == (sd,) and e.action_space.shape == (ad,)
    return envs


def restore(env, obs, nq, nv):
    import numpy as np
    env.reset()
    qpos = np.concatenate([[0.], np.asarray(obs[:nq-1], dtype=np.float64)])
    qvel = np.asarray(obs[nq-1:nq-1+nv], dtype=np.float64)
    env.unwrapped.set_state(qpos, qvel)
    got = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
    return float(np.abs(got-np.asarray(obs, dtype=np.float64)).max()), got


def run(job, out, code):
    import numpy as np
    from asrc_common import read_json, seed_of, sha256_file
    import asrc_data
    from select_worker import load_policy, starts_for
    if job.get('vendor'):
        sys.path.insert(0, job['vendor'])
    sys.path.insert(0, str(code/'otdf_official'))
    sys.path.insert(0, job['algo'])
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = 'cpu'
    if job.get('device', 'cuda') == 'cuda':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(job['gpu'])
        device = 'cuda:0'
    asrc_data.import_d4rl()
    from algo.OTDF import Policy as OTDFPolicy
    from offline_offline.iql_unc import Policy as BCPolicy
    spec = read_json(code/'tasks.json')
    task = next(t for t in spec['tasks'] if t['id'] == job['task'])
    robot = spec['robots'][task['robot']]
    sd, ad, nq, nv = robot['state_dim'], robot['action_dim'], robot['nq'], robot['nv']
    if job.get('target_override'):
        tp = Path(job['target_override'])
    else:
        tp = asrc_data.target_path(task, spec, [Path(p) for p in job['preflight_dirs']], job.get('private_data'))
        assert sha256_file(job['xml']) == job['xml_sha256'], 'XML hash mismatch'
    tar = asrc_data.call_tar_dataset_copy(tp)
    seg = asrc_data.segments_of_raw(tp)[0][:len(tar['observations'])]
    starts, start_info = starts_for(task['id'], tar, seg, np, seed_of)
    N = len(starts)
    horizons = sorted(set([int(v) for v in job['h_star'].values()]+list(job['fixed_horizons'])))
    H = max(horizons+[1])
    envs = make_envs(job, N, sd, ad)
    restore_err = []
    for env, s in zip(envs, starts):
        err, _ = restore(env, s, nq, nv)
        restore_err.append(err)
    max_err = max(restore_err)
    assert max_err <= 1e-6, 'state restoration check failed (max |obs difference| %.3g)' % max_err
    results = {}
    for cand in job['candidates']:
        per_policy = []
        for entry in job['policies'][cand]:
            act = load_policy(torch, entry, OTDFPolicy, BCPolicy, sd, ad, device)
            obs = np.zeros((N, sd), np.float32)
            for i, (env, s) in enumerate(zip(envs, starts)):
                err, got = restore(env, s, nq, nv)
                assert err <= 1e-6
                obs[i] = got
            rewards = np.zeros((N, H))
            alive = np.ones(N, bool)
            end = np.full(N, H)
            fall_end = np.full(N, np.inf)              # step count at which the episode terminated (not time limit)
            for t in range(H):
                with torch.no_grad():
                    a = act(torch.as_tensor(obs, device=device)).cpu().numpy()
                for i in np.flatnonzero(alive):
                    o, r, done, info = envs[i].step(a[i])
                    rewards[i, t] = r
                    obs[i] = o
                    if done:
                        alive[i] = False
                        end[i] = t+1
                        if not info.get('TimeLimit.truncated', False):
                            fall_end[i] = t+1
                if not alive.any():
                    break
            csum = np.concatenate([np.zeros((N, 1)), np.cumsum(rewards, 1)], 1)
            per_policy.append(dict(step=entry['step'], returns={str(h): float(csum[:, min(h, H)].mean()) for h in horizons},
                                   fall_fraction={str(h): float((fall_end <= h).mean()) for h in horizons},
                                   mean_length=float(end.mean())))
        results[cand] = dict(per_policy=per_policy,
                             returns={str(h): sum(p['returns'][str(h)] for p in per_policy)/len(per_policy) for h in horizons})
    for env in envs:
        env.close()
    return dict(task=job['task'], seed=job['seed'], starts=start_info, horizons=horizons, h_star=job['h_star'],
                restore_max_abs_error=max_err, results=results)


if __name__ == '__main__':
    main()
