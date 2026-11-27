"""j013 unit tests (also run locally on CPU with synthetic data).

  a   OTDFTiered with an empty near layer and all layer weights 1 (w_far = 1) vs official OTDF, same seed:
      official VAE pre-training (10000 steps) + 300 train steps; every parameter, optimizer state, scheduler state
      and random state must be byte-identical. a2 (sanity): w_far = 0.5 with the same seed must differ.
  b   rho = 0: official OTDF with proportion 0 (topk of 0 rows) runs 300 steps with finite values; also OTDFTiered
      with proportion 0 and a non-empty near layer.
  c   rho = 1: OTDFTiered with proportion 1 and the rho100T near layer runs 300 steps; also official OTDF with
      proportion 1.
  d   random tier: |rho80R layer| == |rho80T layer|, the random rows lie in D_ret(0.8), are unique, and the draw is
      reproducible.
  layer  per-row weights after one tiered step: target rows 1, far rows exp(c) * w_far, near rows
      min(exp(c'), 1) * w_near; near rows unique when |D_near| >= 64, drawn with replacement when < 64.
usage: python preflight_tests.py <job.json>   (job: test, task, data, cost, steps, device, gpu, output, code)
"""
import json
import os
from pathlib import Path
import random
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
        result = run(job, code)
        write_json(out/'COMPLETED.json', result)
    except BaseException:
        write_json(out/'FAILED.json', dict(time=time.time(), job=job['id'], traceback=scrub(traceback.format_exc())))
        print(scrub(traceback.format_exc()), flush=True)
        sys.exit(1)


class ScalarLog:
    def __init__(self):
        self.rows = []

    def add_scalar(self, tag, value, step):
        self.rows.append([tag, float(value), int(step)])


def setup(job, code):
    if job.get('vendor'):
        sys.path.insert(0, job['vendor'])
    sys.path.insert(0, str(code/'otdf_official'))
    os.environ['D4RL_DATASET_DIR'] = str(code/'private_runtime'/'d4rl_datasets')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda' if job.get('device', 'cuda') == 'cuda' else 'cpu'
    if device == 'cuda':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(job['gpu'])
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    import asrc_data
    asrc_data.import_d4rl()
    import algo.OTDF as otdf_module
    import algo.utils as utils
    assert Path(otdf_module.__file__).resolve() == (code/'otdf_official'/'algo'/'OTDF.py').resolve()
    from asrc_common import verify_official
    verify_official(code)
    return np, torch, device, otdf_module, utils


def load(job, code, np):
    """Training arrays, cost, target segment ids and the Support object (real task or synthetic)."""
    from asrc_common import read_json, sha256_file
    import asrc_data
    from asrc_support import Support
    if job.get('synthetic_data'):
        cfg = job['synthetic_data']
        rng = np.random.RandomState(cfg.get('seed', 7))
        sd, ad = job['state_dim'], job['action_dim']

        def make(k):
            obs = rng.randn(k, sd).astype(np.float32)
            return {'observations': obs, 'actions': np.tanh(rng.randn(k, ad)).astype(np.float32),
                    'next_observations': (obs+.1*rng.randn(k, sd)).astype(np.float32),
                    'rewards': rng.rand(k).astype(np.float32), 'terminals': rng.rand(k) < .01}
        src, tar = make(cfg['source_rows']), make(cfg['target_rows'])
        # make some source rows close to target rows so that the near layer is non-empty
        k = cfg['target_rows']
        src['observations'][:k] = tar['observations']+.01*rng.randn(k, sd).astype(np.float32)
        src['actions'][:k] = tar['actions']
        cost = np.asarray(-rng.rand(cfg['source_rows']).astype(np.float16), dtype=np.float64)   # float16-like ties
        cost[:k] = cost[:k]*.1
        seg = np.repeat(np.arange(5), int(np.ceil(k/5.)))[:k]
        return src, tar, cost, seg, Support(tar, seg, src, cost)
    spec = read_json(code/'tasks.json')
    task = next(t for t in spec['tasks'] if t['id'] == job['task'])
    src, tar, info = asrc_data.load_task_arrays(task, spec, None, [Path(p) for p in job.get('preflight_dirs', [])],
                                                job.get('private_data'))
    if job.get('cost_file'):
        assert sha256_file(job['cost_file']) == job['cost_sha256']
        cost = np.asarray(asrc_data.read_hdf5_dict(job['cost_file'])['cost'])
        cost_kind = 'real'
    else:                                              # OT not available: deterministic stand-in (not for j014)
        rng = np.random.RandomState(11)
        cost = np.asarray(-rng.rand(len(src['observations'])).astype(np.float16), dtype=np.float64)
        cost_kind = 'synthetic'
    job['_cost_kind'] = cost_kind
    tp = asrc_data.target_path(task, spec, [Path(p) for p in job.get('preflight_dirs', [])], job.get('private_data'))
    seg_raw, _ = asrc_data.segments_of_raw(tp)
    seg = seg_raw[:len(tar['observations'])]
    return src, tar, cost, seg, Support(tar, seg, src, cost)


def config_for(job, code, proportion, noreg=False):
    import yaml
    with open(str(code/job['otdf_config']), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['metric'] = 'cosine'
    config.update({'state_dim': job['state_dim'], 'action_dim': job['action_dim'], 'max_action': 1.0, 'weight': True,
                   'proportion': float(proportion), 'noreg': bool(noreg), 'reg_weight': float(job['reg_weight'])})
    return config


def seed_all(np, torch, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def snapshot(agent, np, torch, device):
    """Bytes of every parameter / buffer / optimizer state / scheduler state and the random states."""
    out = {}
    nets = dict(q=agent.q_funcs, target_q=agent.target_q_funcs, v=agent.v_func, actor=agent.policy, vae=agent.vae_policy)
    for n, net in nets.items():
        for k, v in net.state_dict().items():
            out['%s.%s' % (n, k)] = v.detach().cpu().numpy().tobytes()
    opts = dict(q=agent.q_optimizer, v=agent.v_optimizer, actor=agent.policy_optimizer, vae=agent.vae_policy_optimizer)
    for n, opt in opts.items():
        for pid, st in opt.state_dict()['state'].items():
            for k, v in st.items():
                out['opt.%s.%s.%s' % (n, pid, k)] = v.detach().cpu().numpy().tobytes() if hasattr(v, 'detach') else repr(v).encode()
    out['sched'] = repr(sorted(agent.policy_lr_schedule.state_dict().items())).encode()
    out['total_it'] = repr(agent.total_it).encode()
    st = np.random.get_state()
    out['rng.numpy'] = repr((st[0], st[1].tobytes(), st[2:])).encode()
    out['rng.torch'] = torch.get_rng_state().numpy().tobytes()
    if device == 'cuda':
        out['rng.cuda'] = torch.cuda.get_rng_state().numpy().tobytes()
    out['rng.python'] = repr(random.getstate()).encode()
    return out


def train(cls, job, code, np, torch, device, utils, data, proportion, tiers=None, steps=300, seed=2999, noreg=False):
    src, tar, cost = data
    config = config_for(job, code, proportion, noreg)
    seed_all(np, torch, seed)
    agent = cls(config, device)
    src_buf = utils.OTReplayBuffer(job['state_dim'], job['action_dim'], device)
    tar_buf = utils.ReplayBuffer(job['state_dim'], job['action_dim'], device)
    src_buf.convert_D4RL(src)
    tar_buf.convert_D4RL(tar)
    src_buf.cost = cost
    if tiers is not None:
        agent.configure_tiers(src_buf.size, **tiers)
    log = ScalarLog()
    if not config['noreg']:
        agent.train_vae(tar_buf, config['batch_size'], log)
    for _ in range(steps):
        agent.train(src_buf, tar_buf, config['batch_size'], log)
    finite = all(bool(torch.isfinite(p).all()) for net in (agent.q_funcs, agent.v_func, agent.policy, agent.vae_policy)
                 for p in net.parameters())
    return agent, finite, src_buf, tar_buf


def random_tier_check(sup, task_id):
    """Test d: |rho80R layer| == |rho80T layer|; random rows inside D_ret(0.8), unique, reproducible."""
    import numpy as np
    from asrc_common import array_digest
    near = sup.layers('rho80T', task_id)
    rnd = sup.layers('rho80R', task_id)
    rnd2 = sup.layers('rho80R', task_id)
    ret = sup.ret_index(.8)
    inside = bool(np.isin(rnd, ret).all())
    unique = len(np.unique(rnd)) == len(rnd)
    same = bool(np.array_equal(rnd, rnd2))
    return dict(near_rows=int(len(near)), random_rows=int(len(rnd)), same_size=len(near) == len(rnd),
                random_inside_ret=inside, random_unique=unique, reproducible=same,
                overlap_with_near=int(np.isin(rnd, near).sum()), ret_rows=int(len(ret)),
                near_digest=array_digest(near)['sha256'], random_digest=array_digest(rnd)['sha256'],
                passed=(len(near) == len(rnd)) and inside and unique and same)


def run(job, code):
    np, torch, device, otdf_module, utils = setup(job, code)
    from otdf_tiered import OTDFTiered
    from asrc_common import seed_of
    OTDF = otdf_module.OTDF
    t0 = time.time()
    src, tar, cost, seg, sup = load(job, code, np)
    data = (src, tar, cost)
    n = len(src['observations'])
    steps = int(job.get('steps', 300))
    seed = int(job.get('seed', 2999))
    near_seed = seed_of('near-batch', job['task'], seed)
    res = dict(test=job['test'], task=job['task'], steps=steps, seed=seed, device=device, cost_kind=job.get('_cost_kind', 'synthetic'),
               source_rows=n, target_rows=int(len(tar['observations'])), load_seconds=time.time()-t0)
    test = job['test']
    if test == 'a':
        a, fa, _, _ = train(OTDF, job, code, np, torch, device, utils, data, .8, None, steps, seed)
        snap_a = snapshot(a, np, torch, device)
        del a
        b, fb, _, _ = train(OTDFTiered, job, code, np, torch, device, utils, data, .8,
                            dict(near_index=np.zeros(0, np.int64), w_target=1., w_far=1., w_near=.5, near_batch=64,
                                 near_seed=near_seed), steps, seed)
        snap_b = snapshot(b, np, torch, device)
        stats_b = dict(b.tier_stats)
        del b
        diff = sorted(k for k in set(snap_a) | set(snap_b) if snap_a.get(k) != snap_b.get(k))
        c, fc, _, _ = train(OTDFTiered, job, code, np, torch, device, utils, data, .8,
                            dict(near_index=np.zeros(0, np.int64), w_target=1., w_far=.5, w_near=.5, near_batch=64,
                                 near_seed=near_seed), steps, seed)
        snap_c = snapshot(c, np, torch, device)
        del c
        sanity = [k for k in snap_a if k.startswith(('q.', 'v.', 'actor.')) and snap_a[k] != snap_c.get(k)]
        res.update(identical=not diff, compared_entries=len(snap_a), differing=diff[:20], finite=fa and fb and fc,
                   tiered_stats=stats_b, sanity_w_far_half_differs=len(sanity) > 0,
                   passed=(not diff) and fa and fb and fc and len(sanity) > 0)
    elif test == 'b':
        a, fa, _, _ = train(OTDF, job, code, np, torch, device, utils, data, 0., None, steps, seed)
        near = sup.near_index(.8)
        b, fb, _, _ = train(OTDFTiered, job, code, np, torch, device, utils, data, 0.,
                            dict(near_index=near, w_target=1., w_far=.5, w_near=.5, near_batch=64, near_seed=near_seed),
                            steps, seed)
        res.update(official_finite=fa, tiered_finite=fb, tiered_near_rows=int(len(near)), tiered_stats=dict(b.tier_stats),
                   passed=fa and fb)
    elif test == 'c':
        near = sup.near_index(1.)
        a, fa, _, _ = train(OTDFTiered, job, code, np, torch, device, utils, data, 1.,
                            dict(near_index=near, w_target=1., w_far=.5, w_near=.5, near_batch=64, near_seed=near_seed),
                            steps, seed)
        b, fb, _, _ = train(OTDF, job, code, np, torch, device, utils, data, 1., None, steps, seed)
        res.update(tiered_finite=fa, official_finite=fb, near_rows=int(len(near)), tiered_stats=dict(a.tier_stats),
                   passed=fa and fb)
    elif test == 'd':
        res.update(random_tier_check(sup, job['task']))
    elif test == 'layer':
        near = sup.near_index(.8)
        a, fa, src_buf, tar_buf = train(OTDFTiered, job, code, np, torch, device, utils, data, .8,
                                        dict(near_index=near, w_target=1., w_far=.5, w_near=.5, near_batch=64,
                                             near_seed=near_seed), 0, seed)
        a.train(src_buf, tar_buf, 128, None)
        w = a.weight.detach().cpu().numpy().reshape(-1)
        n_src, n_near = int(128*.8), (64 if len(near) else 0)
        far_w, near_w, tar_w = w[:n_src], w[n_src:n_src+n_near], w[n_src+n_near:]
        ok_far = bool(((far_w > 0) & (far_w <= .5+1e-7)).all()) and abs(float(far_w.max())-.5) < 1e-6
        ok_near = bool(((near_w > 0) & (near_w <= .5+1e-7)).all())
        ok_tar = bool((tar_w == 1.).all()) and len(tar_w) == 128
        # near rows: unique draws for |D_near| >= 64, with replacement for a 10-row near set
        rows = [a._near_rows(len(near), 64) for _ in range(50)] if len(near) >= 64 else []
        uniq = all(len(np.unique(r)) == 64 for r in rows)
        small = OTDFTiered(config_for(job, code, .8), device)
        small.configure_tiers(n, np.arange(10), near_seed=1)
        r = small._near_rows(10, 64)
        res.update(weights_rows=int(len(w)), far_ok=ok_far, near_ok=ok_near, target_ok=ok_tar, near_unique_draws=uniq,
                   small_near_with_replacement=bool(len(r) == 64 and r.max() < 10 and small.tier_stats['near_with_replacement_steps'] == 1),
                   near_rows=int(len(near)), finite=fa,
                   passed=fa and ok_far and ok_near and ok_tar and uniq and len(r) == 64 and r.max() < 10)
    else:
        raise ValueError(test)
    res['seconds'] = time.time()-t0
    return res


if __name__ == '__main__':
    main()
