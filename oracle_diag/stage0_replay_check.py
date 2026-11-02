"""Stage 0: can we trust "restore state from observation, step once"?

Checks (per env):
  T  target data (D4RL, seed-0 10% buffer), 5000 random rows, replayed in the
     normal-dynamics env from the observation        -> should be ~0
  T* same rows replayed from the exact infos/qpos,qvel -> separates qvel-clip error
  S  all source rows (ODRL shifted data, call_tar_dataset order) replayed in the
     shifted env                                       -> should be ~0
  X  sensitivity control: the target rows replayed in the shifted env -> should NOT be ~0
"""
import argparse
import time

from common import *  # noqa: F401,F403
from common import (ENVS, RESULTS, TARGET_FRACTION, load_target_buffer, target_subsample_indices,
                    qlearning_raw_indices, load_source_buffer, make_target_env, make_shifted_env,
                    oracle_step, oracle_batch, oracle_batch_exact, is_qvel_clipped, build_bellman_agent,
                    target_obs_std, summarize, dump_json, source_hdf5_path, recover_states,
                    prev_index_from_boundaries)
import numpy as np
import h5py
import d4rl

N_CHECK = 5000
SEED = 0


def errors(S2_hat, S2, R_hat, R, std):
    d = S2_hat - S2
    return {
        'state_l2': np.linalg.norm(d, axis=1),
        'state_l2_std': np.linalg.norm(d / std, axis=1),
        'state_maxabs': np.abs(d).max(axis=1),
        'reward_abs': np.abs(R_hat - R),
    }


def table(err, mask):
    return {k: summarize(v[mask]) for k, v in err.items()}


def run_env(env_name):
    t0 = time.time()
    out = {'env': env_name, 'seed': SEED, 'n_check': N_CHECK, 'target_fraction': TARGET_FRACTION}
    tar_env = make_target_env(env_name)
    sh_env = make_shifted_env(env_name)
    obs_dim = tar_env.observation_space.shape[0]
    act_dim = tar_env.action_space.shape[0]
    out['target_env_mro'] = [c.__module__ + '.' + c.__name__ for c in type(tar_env.unwrapped).__mro__[:3]]
    out['shifted_env_mro'] = [c.__module__ + '.' + c.__name__ for c in type(sh_env.unwrapped).__mro__[:2]]
    out['shifted_xml'] = getattr(sh_env.unwrapped, '_xml_file', None) or getattr(sh_env.unwrapped, 'fullpath', None)
    out['frame_skip'] = {'target': tar_env.unwrapped.frame_skip, 'shifted': sh_env.unwrapped.frame_skip}
    out['dt'] = {'target': tar_env.unwrapped.dt, 'shifted': sh_env.unwrapped.dt}

    # ---- target buffer exactly as train.py builds it, plus index mapping back to the hdf5
    tar_buf, full_size = load_target_buffer(env_name, SEED)
    sub_idx = target_subsample_indices(full_size, TARGET_FRACTION, SEED, obs_dim, act_dim)
    raw_idx, ds = qlearning_raw_indices(env_name)
    assert len(raw_idx) == full_size, (len(raw_idx), full_size)
    raw_of_buf = raw_idx[sub_idx]
    assert np.array_equal(ds['observations'][raw_of_buf].astype(np.float32), tar_buf.state.astype(np.float32))
    assert np.array_equal(ds['actions'][raw_of_buf].astype(np.float32), tar_buf.action.astype(np.float32))
    out['target_buffer_size'] = int(tar_buf.size)
    out['target_full_size'] = int(full_size)

    agent = build_bellman_agent(env_name, SEED, tar_buf)
    std = target_obs_std(agent, obs_dim)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(tar_buf.size, size=N_CHECK, replace=False)
    raw = raw_of_buf[pick]
    S = tar_buf.state[pick].astype(np.float64)
    A = tar_buf.action[pick].astype(np.float64)
    R = tar_buf.reward[pick].ravel().astype(np.float64)
    ND = tar_buf.not_done[pick].ravel()
    S2_true = ds['next_observations'][raw].astype(np.float64)  # correct even at terminals
    S2_buf = tar_buf.next_state[pick].astype(np.float64)
    out['target_buf_vs_hdf5_next_obs_mismatch'] = int((np.abs(S2_buf - S2_true).max(1) > 1e-6).sum())

    clipped = is_qvel_clipped(tar_env, env_name, S)
    clipped_next = is_qvel_clipped(tar_env, env_name, S2_true)

    # T: from observation
    R_o, S2_o, D_o = oracle_batch(tar_env, S, A)
    err_T = errors(S2_o, S2_true, R_o, R, std)
    # T*: recover the true (unclipped) internal state along the raw D4RL trajectory.
    # infos/qvel in D4RL is itself clipped, so it cannot be used directly.
    n = len(S)
    raw_obs = ds['observations']
    raw_act = ds['actions']
    ep_end = np.logical_or(ds['terminals'], ds['timeouts'])
    raw_clip = lambda r: bool(is_qvel_clipped(tar_env, env_name, raw_obs[r:r + 1])[0])
    needed = {}
    for r in raw:
        r = int(r)
        cur = r
        depth = 0
        while cur not in needed:
            needed[cur] = None
            if not raw_clip(cur) or cur == 0 or ep_end[cur - 1] or depth > 200:
                break
            cur -= 1
            depth += 1
    rows = np.array(sorted(needed), dtype=np.int64)
    pos = {int(r): k for k, r in enumerate(rows)}
    prev = np.array([pos.get(int(r) - 1, -1) if (r > 0 and not ep_end[r - 1]) else -1 for r in rows])
    c_obs = raw_obs[rows].astype(np.float64)
    c_clip = is_qvel_clipped(tar_env, env_name, c_obs)
    QP, QV, st, verr = recover_states(tar_env, c_obs, raw_act[rows].astype(np.float64), prev, c_clip)
    sel = np.array([pos[int(r)] for r in raw])
    R_x, S2_x, D_x = oracle_batch_exact(tar_env, QP[sel], QV[sel], A)
    status_t = st[sel]
    err_Tx = errors(S2_x, S2_true, R_x, R, std)
    # X: sensitivity control in shifted env
    R_c, S2_c, D_c = oracle_batch(sh_env, S, A)
    err_X = errors(S2_c, S2_true, R_c, R, std)

    allm = np.ones(n, dtype=bool)
    out['target_check'] = {
        'frac_qvel_clipped_input': float(clipped.mean()),
        'frac_qvel_clipped_next': float(clipped_next.mean()),
        'done_agreement_obs_replay': float(((~D_o) == (ND > 0.5)).mean()),
        'from_obs': {'all': table(err_T, allm), 'not_clipped': table(err_T, ~clipped), 'clipped': table(err_T, clipped)},
        'recovery_status_counts': {'exact_from_obs': int((status_t == 0).sum()), 'recovered': int((status_t == 1).sum()), 'unrecoverable': int((status_t == 2).sum())},
        'from_recovered_state': {'all': table(err_Tx, allm), 'recovered_only': table(err_Tx, status_t == 1), 'excluding_unrecoverable': table(err_Tx, status_t != 2)},
        'done_agreement_recovered_replay': float(((~D_x) == (ND > 0.5)).mean()),
        'control_target_rows_in_shifted_env': {'all': table(err_X, allm), 'not_clipped': table(err_X, ~clipped)},
    }

    # ---- S: source data in shifted env
    src_buf = load_source_buffer(env_name)
    with h5py.File(source_hdf5_path(env_name), 'r') as f:
        h_obs = f['observations'][:]
        h_next = f['next_observations'][:]
    m = src_buf.size
    assert np.array_equal(h_obs[:m].astype(np.float32), src_buf.state.astype(np.float32))
    boundary = np.abs(h_next[:m] - src_buf.next_state).max(1) > 1e-6  # call_tar_dataset uses obs[i+1]
    Ss = src_buf.state.astype(np.float64)
    As = src_buf.action.astype(np.float64)
    Rs = src_buf.reward.ravel().astype(np.float64)
    NDs = src_buf.not_done.ravel()
    S2s_true = h_next[:m].astype(np.float64)
    S2s_buf = src_buf.next_state.astype(np.float64)
    clipped_s = is_qvel_clipped(sh_env, env_name, Ss)
    R_s, S2_s, D_s = oracle_batch(sh_env, Ss, As)
    err_S = errors(S2_s, S2s_true, R_s, Rs, std)
    err_S_buf = errors(S2_s, S2s_buf, R_s, Rs, std)
    ep_start = np.zeros(m, dtype=bool)
    ep_start[0] = True
    ep_start[1:] = boundary[:-1]
    prev_s = prev_index_from_boundaries(m, ep_start)
    QPs, QVs, st_s, _ = recover_states(sh_env, Ss, As, prev_s, clipped_s)
    R_sx, S2_sx, D_sx = oracle_batch_exact(sh_env, QPs, QVs, As)
    err_Sx = errors(S2_sx, S2s_true, R_sx, Rs, std)
    good = ~boundary
    out['source_check'] = {
        'n': int(m),
        'n_boundary_rows_obs_i_plus_1_ne_next_obs': int(boundary.sum()),
        'boundary_rows': np.where(boundary)[0].tolist(),
        'boundary_rows_not_done': NDs[boundary].tolist(),
        'frac_qvel_clipped_input': float(clipped_s.mean()),
        'done_agreement': float(((~D_s) == (NDs > 0.5)).mean()),
        'vs_hdf5_next_observations': {'all': table(err_S, np.ones(m, bool)), 'not_clipped': table(err_S, ~clipped_s), 'clipped': table(err_S, clipped_s)},
        'recovery_status_counts': {'exact_from_obs': int((st_s == 0).sum()), 'recovered': int((st_s == 1).sum()), 'unrecoverable': int((st_s == 2).sum())},
        'recovered_vs_hdf5_next_observations': {'all': table(err_Sx, np.ones(m, bool)), 'recovered_only': table(err_Sx, st_s == 1), 'excluding_unrecoverable': table(err_Sx, st_s != 2)},
        'done_agreement_recovered': float(((~D_sx) == (NDs > 0.5)).mean()),
        'vs_buffer_next_state_excluding_boundary': table(err_S_buf, good),
        'vs_buffer_next_state_boundary_only': table(err_S_buf, boundary),
    }
    out['elapsed_sec'] = time.time() - t0
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--envs', nargs='*', default=ENVS)
    args = ap.parse_args()
    allres = {}
    for e in args.envs:
        print('=== stage0', e, flush=True)
        allres[e] = run_env(e)
        dump_json(allres[e], RESULTS / 'stage0_{}.json'.format(e))
        tc = allres[e]['target_check']
        sc = allres[e]['source_check']
        print(' target from_obs state_l2_std', tc['from_obs']['all']['state_l2_std'])
        print(' target from_obs reward_abs', tc['from_obs']['all']['reward_abs'])
        print(' target recovered state_l2_std', tc['from_recovered_state']['all']['state_l2_std'], tc['recovery_status_counts'])
        print(' target recovered reward_abs', tc['from_recovered_state']['all']['reward_abs'])
        print(' control shifted  state_l2_std', tc['control_target_rows_in_shifted_env']['all']['state_l2_std'])
        print(' source           state_l2_std', sc['vs_hdf5_next_observations']['all']['state_l2_std'])
        print(' source           reward_abs', sc['vs_hdf5_next_observations']['all']['reward_abs'])
        print(' source recovered state_l2_std', sc['recovered_vs_hdf5_next_observations']['all']['state_l2_std'], sc['recovery_status_counts'])
        print(' source recovered reward_abs', sc['recovered_vs_hdf5_next_observations']['all']['reward_abs'])
        print(' elapsed', allres[e]['elapsed_sec'], flush=True)
