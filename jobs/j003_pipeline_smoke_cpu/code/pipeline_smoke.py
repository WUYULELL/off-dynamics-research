"""j003 pipeline smoke test (CPU only, no research result).

Checks, inside the auto-job environment, everything a later training job needs:
  1. MuJoCo runtime ($HOME/.mujoco210/mujoco210) and a private copy of mujoco_py
     (copied into this job folder, the installed package is never imported or written);
  2. the ODRL target archive (SHA-256), unpacked into this job's own data/ folder,
     with an inventory of every file it contains;
  3. the D4RL source datasets and the frozen learner files against the transfer MANIFEST;
  4. per task (the three V21 tasks): load target/source data, descriptive segment-return
     statistics, build the target environment from the V21 XML, a few IQL (target+source
     batch) and BC updates on CPU, and one evaluation episode.
Smoke seed 2999 only; returns of the smoke policy are not reported (only finiteness and length).
Reads the transfer project and the archive read-only; writes only under this job's code/ folder.
Exit code 1 if any check fails.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs'
HOME = Path.home()
PROJECT = HOME/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
ALGO = PROJECT/'source_snapshot/algo'
ARCHIVE = HOME/'odrl_target_20261115.zip'
ARCHIVE_SHA = 'df9c2b43c0a1f479c2ebeac1a3bbf156a95fa8aaedc6e92d36443baea85ae6fc'
MUJOCO = HOME/'.mujoco210/mujoco210'
SEED = 2999
IQL_STEPS = 1000
BC_STEPS = 500
KEYS = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')
ROBOTS = {'hopper': dict(env=('gym.envs.mujoco.hopper_v3', 'HopperEnv'), source_task='hopper-morph', sd=11, ad=3),
          'walker2d': dict(env=('gym.envs.mujoco.walker2d_v3', 'Walker2dEnv'), source_task='walker2d-morph', sd=17, ad=6)}
TASKS = [
    dict(id='hopper-kinematic-legjnt-medium', robot='hopper',
         target_file='hopper_kinematic_legjnt_medium_medium.hdf5',
         target_sha256='82f4414e6acb88523ed809b60b56838b6b68da603d75ee72eb79362515cf00d0',
         xml='hopper_kinematic_legjnt_medium.xml',
         xml_sha256='11e5861cd1fcd4142418978911b95240001f142282f3cacdacfe99bf3a1209a8',
         score_min=-26.3360015397715, score_max=3234.3),
    dict(id='walker2d-kinematic-thighjnt-medium', robot='walker2d',
         target_file='walker2d_kinematic_thighjnt_medium_medium.hdf5',
         target_sha256='cae31125683286271a798d52bde1f0e2a0650baf527e5f5ff490d6d7cde03a07',
         xml='walker2d_kinematic_thighjnt_medium.xml',
         xml_sha256='bf4797354c7adc42c86572e1365b08ae6c7e22d2e75c4f4122964e196036b699',
         score_min=10.079455055289959, score_max=4592.3),
    dict(id='walker2d-kinematic-thighjnt-hard', robot='walker2d',
         target_file='walker2d_kinematic_thighjnt_hard_medium.hdf5',
         target_sha256='16b792230e8018a1028ba636d79a47797448e41aa6c60af5159acef5cebe0c58',
         xml='walker2d_kinematic_thighjnt_hard.xml',
         xml_sha256='cabadec6f494e02ae4d9ff3bc37d4f81401f8cb87a8d54a8fdca177f91724458',
         score_min=10.079455055289959, score_max=4592.3),
]

checks = []
summary = dict(job='j003_pipeline_smoke_cpu', seed=SEED, started=time.strftime('%Y-%m-%d %H:%M:%S'),
               python=sys.version.split()[0], executable=sys.executable, checks=checks)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save():
    tmp = OUT/'summary.json.tmp'
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(str(tmp), str(OUT/'summary.json'))


def check(name, fn):
    t = time.time()
    try:
        detail = fn()
        checks.append(dict(name=name, ok=True, seconds=round(time.time()-t, 2), detail=detail))
        print('[OK]', name, json.dumps(detail, ensure_ascii=False, default=str)[:600], flush=True)
        return True
    except BaseException:
        tb = traceback.format_exc()
        checks.append(dict(name=name, ok=False, seconds=round(time.time()-t, 2), error=tb[-3000:]))
        print('[FAIL]', name, '\n', tb, flush=True)
        return False
    finally:
        save()


def segment_stats(obs, nxt, rew, term, timeouts, lo, hi):
    """Segments cut at terminals, timeouts and discontinuities (same definition as V21); normalised returns."""
    import numpy as np
    cont = np.zeros(len(rew), bool)
    cont[:-1] = np.all(np.abs(nxt[:-1]-obs[1:]) <= 1e-5, axis=1)
    end = term | timeouts | ~cont
    end[-1] = True
    seg = np.concatenate([[0], np.cumsum(end[:-1])]).astype(np.int64)
    ret = np.bincount(seg, weights=rew)
    length = np.bincount(seg)
    return seg, 100*(ret-lo)/(hi-lo), length


def main():
    OUT.mkdir(exist_ok=True)
    os.chdir(str(HERE))

    def frozen():
        spec = json.loads((HERE/'FROZEN.json').read_text(encoding='utf-8'))
        bad = [n for n, d in spec['files'].items() if sha(HERE/n) != d]
        assert not bad, 'hash mismatch: %s' % bad
        return dict(files=len(spec['files']))
    check('frozen_package', frozen)

    def env_vars():
        keys = ['MUJOCO_PY_MUJOCO_PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'MUJOCO_PY_FORCE_CPU',
                'OMP_NUM_THREADS', 'AUTO_JOB_ID']
        d = {k: os.environ.get(k) for k in keys}
        assert (MUJOCO/'bin').is_dir(), 'missing %s' % (MUJOCO/'bin')
        d['mujoco_bin'] = sorted(p.name for p in (MUJOCO/'bin').iterdir())[:30]
        d['nproc'] = os.cpu_count()
        return d
    check('mujoco_runtime_present', env_vars)

    state = {}

    def binding():
        from importlib import metadata
        original = Path(metadata.distribution('mujoco-py').locate_file('mujoco_py')).resolve()
        assert (original/'__init__.py').is_file()
        vendor = HERE/'private_runtime'
        if not (vendor/'mujoco_py').is_dir():
            vendor.mkdir(exist_ok=True)
            shutil.copytree(str(original), str(vendor/'mujoco_py'))
        sys.path.insert(0, str(vendor))
        t = time.time()
        import mujoco_py
        loc = Path(mujoco_py.__file__).resolve()
        assert vendor.resolve() in loc.parents, 'imported %s, not the private copy' % loc
        state['mujoco_ok'] = True
        return dict(original=str(original), imported_from=str(loc), import_seconds=round(time.time()-t, 1),
                    version=getattr(mujoco_py, '__version__', None))
    check('mujoco_py_private_copy_import', binding)

    def archive():
        assert ARCHIVE.is_file(), 'missing %s' % ARCHIVE
        digest = sha(ARCHIVE)
        assert digest == ARCHIVE_SHA, 'archive hash %s' % digest
        data = HERE/'data'
        inventory = []
        with zipfile.ZipFile(str(ARCHIVE)) as z:
            for info in z.infolist():
                inventory.append(dict(name=info.filename, bytes=info.file_size))
                if info.filename.endswith('.hdf5') and '/raw/' in info.filename:
                    data.mkdir(exist_ok=True)
                    dst = data/Path(info.filename).name
                    assert not dst.exists() or sha(dst) == hashlib.sha256(z.read(info)).hexdigest(), 'duplicate name %s' % dst.name
                    dst.write_bytes(z.read(info))
        hashes = {p.name: sha(p) for p in sorted(data.glob('*.hdf5'))}
        for task in TASKS:
            assert hashes.get(task['target_file']) == task['target_sha256'], task['target_file']
            assert sha(HERE/'assets'/task['xml']) == task['xml_sha256'], task['xml']
        (OUT/'odrl_archive_inventory.json').write_text(json.dumps(dict(entries=inventory, extracted_sha256=hashes),
                                                                 indent=2), encoding='utf-8')
        state['archive_ok'] = True
        return dict(entries=len(inventory), extracted=hashes)
    check('odrl_target_archive', archive)

    def manifest():
        files = {str((PROJECT/e['path']).resolve()): e for e in json.loads((PROJECT/'MANIFEST.json').read_text())['files']}
        spec = json.loads((PROJECT/'cluster/paths.example.json').read_text())['tasks']
        paths = [ALGO/'offline_offline/iql_dyn.py', ALGO/'offline_offline/iql_unc.py']
        paths += [PROJECT/spec[ROBOTS[r]['source_task']]['source_dataset'] for r in ROBOTS]
        out = {}
        for p in paths:
            e = files[str(p.resolve())]
            d = sha(p)
            assert d == e['sha256'] and p.stat().st_size == e['bytes'], str(p)
            out[str(p.relative_to(PROJECT))] = d
        state['source'] = {r: PROJECT/spec[ROBOTS[r]['source_task']]['source_dataset'] for r in ROBOTS}
        return out
    check('transfer_manifest', manifest)

    for task in TASKS:
        check('task_' + task['id'], lambda task=task: run_task(task, state))

    ok = all(c['ok'] for c in checks)
    summary['all_passed'] = ok
    summary['ended'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save()
    lines = ['# j003_pipeline_smoke_cpu 报告', '', '全部通过：%s' % ok, '']
    for c in checks:
        lines.append('- %s %s（%.1f s）' % ('通过' if c['ok'] else '失败', c['name'], c['seconds']))
        if not c['ok']:
            lines.append('  - 错误末尾：`%s`' % c['error'].strip().splitlines()[-1][:300])
    lines += ['', '详细数据见 summary.json 和 odrl_archive_inventory.json；冒烟策略的回报不报告。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return 0 if ok else 1


def run_task(task, state):
    assert state.get('archive_ok') and state.get('mujoco_ok') and 'source' in state, 'prerequisite check failed'
    import numpy as np
    import h5py
    import torch
    sys.path.insert(0, str(ALGO))
    from offline_offline import iql_dyn, iql_unc
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '1'))))
    robot = ROBOTS[task['robot']]
    sd, ad = robot['sd'], robot['ad']
    lo, hi = task['score_min'], task['score_max']
    detail = {}

    def load(path, drop_timeouts):
        with h5py.File(str(path), 'r') as f:
            arr = [np.asarray(f[k], dtype=np.float32) for k in KEYS]
            timeouts = np.asarray(f['timeouts']).reshape(-1).astype(bool) if 'timeouts' in f else np.zeros(len(arr[0]), bool)
        n = len(arr[0])
        assert all(len(a) == n and np.isfinite(a).all() for a in arr)
        assert arr[0].shape == (n, sd) and arr[1].shape == (n, ad) and arr[2].shape == (n, sd)
        seg, norm, length = segment_stats(arr[0].astype(np.float64), arr[2].astype(np.float64),
                                          arr[3].astype(np.float64).reshape(-1), arr[4].reshape(-1).astype(bool),
                                          timeouts, lo, hi)
        if drop_timeouts:
            keep = ~timeouts
            arr = [a[keep] for a in arr]
            seg = seg[keep]
        arr[3] = arr[3].reshape(-1, 1)
        arr[4] = 1.-arr[4].reshape(-1, 1)
        return arr, norm, length, seg, int(timeouts.sum())

    target, t_norm, t_len, _, _ = load(HERE/'data'/task['target_file'], False)
    source, s_norm, s_len, s_seg, s_to = load(state['source'][task['robot']], True)
    assert len(target[0]) == 5000
    thr = float(t_norm.mean())
    kept_rows = int((s_norm[s_seg] >= thr).sum())
    detail['data'] = dict(target_rows=len(target[0]), target_segments=len(t_norm),
                          target_norm_return_mean=round(thr, 3), target_norm_return_std=round(float(t_norm.std()), 3),
                          target_segment_length_mean=round(float(t_len.mean()), 1),
                          source_rows_after_timeout_drop=len(source[0]), source_timeout_rows=s_to,
                          source_segments=len(s_norm), source_norm_return_mean=round(float(s_norm.mean()), 3),
                          source_segments_at_or_above_target_mean=int((s_norm >= thr).sum()),
                          source_rows_kept_by_V21_quality_filter=kept_rows)

    import gym
    cls = getattr(__import__(robot['env'][0], fromlist=[robot['env'][1]]), robot['env'][1])
    env = gym.wrappers.TimeLimit(cls(xml_file=str(HERE/'assets'/task['xml'])), max_episode_steps=1000)
    assert env.unwrapped.frame_skip == 4 and abs(float(env.unwrapped.model.opt.timestep)-.002) < 1e-12
    assert env.observation_space.shape == (sd,) and env.action_space.shape == (ad,)
    env.seed(SEED)
    env.action_space.seed(SEED)
    s, done, n = env.reset(), False, 0
    while not done:
        s, r, done, _ = env.step(env.action_space.sample())
        assert np.isfinite(s).all() and math.isfinite(r)
        n += 1
    detail['random_episode_length'] = n

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cpu'
    cfg = dict(state_dim=sd, action_dim=ad, hidden_sizes=256, max_action=1., gamma=.99, tau=.005, update_interval=2,
               lam=.7, temp=10., critic_lr=.0003, actor_lr=.0003, max_step=IQL_STEPS, seed=SEED,
               source_budget=1., source_loss_weight=1.)
    model = iql_dyn.IQL_DYN(cfg, device)
    rng = np.random.RandomState(SEED)

    def batch(arr):
        ix = rng.randint(len(arr[0]), size=64)
        return tuple(torch.as_tensor(a[ix], dtype=torch.float32, device=device) for a in arr)

    t0 = time.time()
    for _ in range(IQL_STEPS):          # the V21 MIXALL update (target + source batch in V, Q and policy)
        model.total_it += 1
        ts, ta, tn, tr, tm = batch(target)
        ss, sa, sn, sr, sm = batch(source)
        tv, adv, _ = model.v_loss_per_sample(ts, ta)
        sv, s_adv, _ = model.v_loss_per_sample(ss, sa)
        v_loss = tv.mean()+sv.mean()
        model.v_optimizer.zero_grad(); v_loss.backward(); model.v_optimizer.step()
        tq, _, _ = model.q_loss_per_sample(ts, ta, tr, tn, tm)
        sq, _, _ = model.q_loss_per_sample(ss, sa, sr, sn, sm)
        q_loss = tq.mean()+sq.mean()
        model.q_optimizer.zero_grad(); q_loss.backward(); model.q_optimizer.step()
        model.update_target()
        for p in model.q_funcs.parameters():
            p.requires_grad = False
        pi = model.policy_loss_per_sample(adv, ts, ta).mean()+model.policy_loss_per_sample(s_adv, ss, sa).mean()
        model.policy_optimizer.zero_grad(); pi.backward(); model.policy_optimizer.step()
        model.policy_lr_schedule.step()
        for p in model.q_funcs.parameters():
            p.requires_grad = True
    iql_sec = time.time()-t0
    losses = [float(v_loss), float(q_loss), float(pi)]
    nets = [model.policy, model.v_func, model.q_funcs, model.target_q_funcs]
    assert all(math.isfinite(x) for x in losses) and all(bool(torch.isfinite(p).all()) for m in nets for p in m.parameters())

    policy = iql_unc.Policy(sd, ad, 1., 256).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=.0003)
    t0 = time.time()
    for _ in range(BC_STEPS):
        b, sb = batch(target), batch(source)
        loss = policy.bc_loss(b[0], b[1]).mean(dim=1, keepdim=True).mean()+policy.bc_loss(sb[0], sb[1]).mean(dim=1, keepdim=True).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    bc_sec = time.time()-t0
    assert math.isfinite(float(loss))

    env.seed(SEED+1)
    s, done, n, ret = env.reset(), False, 0, 0.
    while not done:
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32).reshape(1, -1)
            a = torch.tanh(model.policy.network(x).chunk(2, dim=1)[0]).squeeze(0).numpy()
        s, r, done, _ = env.step(a)
        ret += float(r)
        n += 1
    assert math.isfinite(ret)
    env.close()
    detail['cpu_seconds_per_1000_IQL_mix_updates'] = round(iql_sec/IQL_STEPS*1000, 2)
    detail['cpu_seconds_per_1000_BC_mix_updates'] = round(bc_sec/BC_STEPS*1000, 2)
    detail['eval_episode_length'] = n
    detail['eval_return_finite'] = True
    return detail


if __name__ == '__main__':
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.exit(code)
