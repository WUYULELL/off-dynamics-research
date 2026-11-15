"""V17 worker: one target-only training run (IQL or BC) with an optional small-data regularizer.

Network classes and the IQL loss functions are the frozen ones in the transfer
project (source_snapshot/algo/offline_offline/iql_dyn.py, iql_unc.py). The IQL
update order is the frozen R13/R14 Learner.step with target data only; BC is the
frozen V9 Actor.step. Hyperparameters equal the earlier T and BC-all runs.

Variants:
  base    no regularizer
  wd      AdamW with decoupled weight decay (every optimizer)
  sparse  SNIP-style global magnitude-times-gradient masks on every Linear weight
          of every network, refreshed every `mask_every` steps until `mask_until`
          (after Sparse-Reg, arXiv 2506.17155; details the paper leaves open are
          fixed in config.json and were not checked against the authors' code)
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

KEYS = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def seed_of(purpose, task, seed):
    text = 'V17-20261115|%s|%d|%s' % (task, seed, purpose)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


def run(job, out):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(job['gpu'])
    sys.path.insert(0, job['vendor'])
    sys.path.insert(0, job['algo'])
    import numpy as np
    import torch
    import h5py
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    from offline_offline import iql_dyn, iql_unc
    assert Path(iql_dyn.__file__).resolve().parent == (Path(job['algo'])/'offline_offline').resolve()
    from torch.nn.utils import prune
    device = 'cuda:0'
    task, learner, variant, steps = job['task'], job['learner'], job['variant'], job['steps']
    sd, ad = job['state_dim'], job['action_dim']

    with h5py.File(job['target_dataset'], 'r') as f:
        arrays = [np.asarray(f[k], dtype=np.float32) for k in KEYS]
    arrays[3] = arrays[3].reshape(-1, 1)
    assert np.isin(arrays[4], [0, 1]).all()
    arrays[4] = 1.-arrays[4].reshape(-1, 1)
    n = len(arrays[0])
    assert n == 5000 and all(len(a) == n and np.isfinite(a).all() for a in arrays)
    assert arrays[0].shape == (n, sd) and arrays[1].shape == (n, ad) and arrays[2].shape == (n, sd)

    s = seed_of('initialization', task, job['seed'])
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if learner == 'IQL':
        config = dict(state_dim=sd, action_dim=ad, hidden_sizes=256, max_action=1., gamma=.99, tau=.005,
                      update_interval=2, lam=.7, temp=10., critic_lr=.0003, actor_lr=.0003, max_step=steps,
                      seed=job['seed'], source_budget=1., source_loss_weight=1.)
        model = iql_dyn.IQL_DYN(config, device)
        policy = model.policy
        nets = dict(policy=model.policy, V=model.v_func, Q=model.q_funcs)
        masked = dict(nets, target_Q=model.target_q_funcs)
    else:
        assert learner == 'BC'
        config = dict(lr=.0003, hidden_sizes=256, cosine_T_max=steps, loss='action-dimension mean squared error, then batch mean')
        model = None
        policy = iql_unc.Policy(sd, ad, 1., 256).to(device)
        nets = dict(policy=policy)
        masked = dict(nets)

    def linears(net):
        return [m for m in net.modules() if isinstance(m, torch.nn.Linear)]

    if variant == 'sparse':
        for net in masked.values():
            for layer in linears(net):
                prune.custom_from_mask(layer, 'weight', torch.ones_like(layer.weight))

    def optimizer(net):
        params = [p for p in net.parameters()]
        if variant == 'wd':
            return torch.optim.AdamW(params, lr=.0003, weight_decay=job['weight_decay'])
        return torch.optim.Adam(params, lr=.0003)

    # Fresh optimizers for every variant, built the same way, before the first update.
    if learner == 'IQL':
        model.q_optimizer = optimizer(model.q_funcs)
        model.v_optimizer = optimizer(model.v_func)
        model.policy_optimizer = optimizer(model.policy)
        model.policy_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(model.policy_optimizer, steps)
    else:
        bc_optimizer = optimizer(policy)
        bc_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(bc_optimizer, steps)

    def digest():
        h = hashlib.sha256()
        for net in masked.values():
            for value in net.state_dict().values():
                h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def finite():
        return all(bool(torch.isfinite(p).all()) for net in masked.values() for p in net.parameters())

    write(out/'config.json', dict(job=job, learner_config=config, batch=64, observation_scaling='raw', reward_scaling=1,
        threads=1, deterministic_algorithms=True, TF32=True, target_dataset_sha256=sha(job['target_dataset']),
        xml_sha256=sha(job['xml']), iql_dyn_sha256=sha(iql_dyn.__file__), iql_unc_sha256=sha(iql_unc.__file__),
        initialization_sha256=digest(), torch=torch.__version__))

    sample_rng = np.random.RandomState(seed_of('target_sample', task, job['seed']))
    mask_rng = np.random.RandomState(seed_of('mask_sample', task, job['seed']))
    stream = hashlib.sha256()

    def tensors(ix):
        return tuple(torch.as_tensor(np.asarray(a[ix]), dtype=torch.float32, device=device) for a in arrays)

    def sample():
        ix = sample_rng.randint(n, size=64)
        stream.update(ix.astype('<i8').tobytes())
        return tensors(ix)

    def iql_step(batch):
        # Frozen R13/R14 Learner.step, target data only.
        model.total_it += 1
        ts, ta, tn, tr, tm = batch
        tv, adv, _ = model.v_loss_per_sample(ts, ta)
        v_loss = tv.mean()
        model.v_optimizer.zero_grad()
        v_loss.backward()
        model.v_optimizer.step()
        tq, _, _ = model.q_loss_per_sample(ts, ta, tr, tn, tm)
        q_loss = tq.mean()
        model.q_optimizer.zero_grad()
        q_loss.backward()
        model.q_optimizer.step()
        model.update_target()
        for p in model.q_funcs.parameters():
            p.requires_grad = False
        pi_loss = model.policy_loss_per_sample(adv, ts, ta).mean()
        model.policy_optimizer.zero_grad()
        pi_loss.backward()
        model.policy_optimizer.step()
        model.policy_lr_schedule.step()
        for p in model.q_funcs.parameters():
            p.requires_grad = True
        return float(v_loss.detach()), float(q_loss.detach()), float(pi_loss.detach())

    def bc_step(batch):
        # Frozen V9 Actor.step.
        loss = policy.bc_loss(batch[0], batch[1]).mean(dim=1, keepdim=True).mean()
        bc_optimizer.zero_grad()
        loss.backward()
        bc_optimizer.step()
        bc_schedule.step()
        return (float(loss.detach()),)

    def refresh(net, loss_fn):
        """Global top-k of |weight * dLoss/dweight| over all Linear weights of one network, computed with
        every connection of that network active. Biases are never masked. Pruned weights keep their last
        value, so they can return at a later refresh."""
        layers = linears(net)
        for layer in layers:
            layer.weight_mask.fill_(1.)
        grads = torch.autograd.grad(loss_fn(), [layer.weight_orig for layer in layers])
        scores = [(layer.weight_orig.detach()*g).abs() for layer, g in zip(layers, grads)]
        flat = torch.cat([x.flatten() for x in scores])
        k = max(1, int(round(job['keep_fraction']*flat.numel())))
        threshold = torch.topk(flat, k, largest=True, sorted=True).values[-1]
        for layer, score in zip(layers, scores):
            mask = (score >= threshold).float()
            floor = max(1, int(math.ceil(job['layer_floor_fraction']*score.numel())))
            if int(mask.sum()) < floor:      # keep a layer from being cut off entirely
                top = torch.topk(score.flatten(), floor, largest=True, sorted=True).values[-1]
                mask = (score >= top).float()
            layer.weight_mask.copy_(mask)
        return [float(layer.weight_mask.mean()) for layer in layers]

    mask_log = []

    def refresh_all(step):
        batch = tensors(mask_rng.randint(n, size=job['mask_batch']))
        s_, a_, n_, r_, m_ = batch
        density = {}
        if learner == 'IQL':
            for p in model.q_funcs.parameters():
                assert p.requires_grad
            density['V'] = refresh(model.v_func, lambda: model.v_loss_per_sample(s_, a_)[0].mean())
            density['Q'] = refresh(model.q_funcs, lambda: model.q_loss_per_sample(s_, a_, r_, n_, m_)[0].mean())
            for target, current in zip(linears(model.target_q_funcs), linears(model.q_funcs)):
                target.weight_mask.copy_(current.weight_mask)
            with torch.no_grad():
                _, adv, _ = model.v_loss_per_sample(s_, a_)
            density['policy'] = refresh(model.policy, lambda: model.policy_loss_per_sample(adv, s_, a_).mean())
        else:
            density['policy'] = refresh(policy, lambda: policy.bc_loss(s_, a_).mean(dim=1, keepdim=True).mean())
        mask_log.append(dict(step=step, density=density))

    def act(state):
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
            return torch.tanh(policy.network(x).chunk(2, dim=1)[0]).squeeze(0).cpu().numpy()

    import gym
    module_name, class_name = job['env_module'], job['env_class']
    constructor = getattr(__import__(module_name, fromlist=[class_name]), class_name)
    env = gym.wrappers.TimeLimit(constructor(xml_file=job['xml']), max_episode_steps=1000)
    assert env.unwrapped.frame_skip == 4 and abs(float(env.unwrapped.model.opt.timestep)-.002) < 1e-12

    def evaluate(step):
        saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        returns, lengths, seeds = [], [], []
        try:
            for episode in range(10):
                value = 500000+10*(job['seed']-job['seed_origin'])+episode
                seeds.append(value)
                env.seed(value)
                state, done, ret, length = env.reset(), False, 0., 0
                while not done:
                    state, reward, done, _ = env.step(act(state))
                    ret += float(reward)
                    length += 1
                returns.append(ret)
                lengths.append(length)
        finally:
            random.setstate(saved[0])
            np.random.set_state(saved[1])
            torch.set_rng_state(saved[2])
            torch.cuda.set_rng_state_all(saved[3])
        scores = (np.asarray(returns)-job['score_min'])/(job['score_max']-job['score_min'])*100.
        assert np.isfinite(scores).all()
        return dict(step=step, raw_returns=returns, scores=scores.tolist(), mean=float(scores.mean()),
                    episode_lengths=lengths, env_seeds=seeds, steps=int(sum(lengths)))

    started = time.monotonic()
    if variant == 'sparse':
        refresh_all(0)
    update = iql_step if learner == 'IQL' else bc_step
    evaluations = 0
    try:
        for step in range(1, steps+1):
            losses = update(sample())
            if variant == 'sparse' and step % job['mask_every'] == 0 and step <= job['mask_until']:
                refresh_all(step)
            if step % 1000 == 0 or step == steps:
                assert all(math.isfinite(x) for x in losses) and finite(), 'nonfinite at step %d' % step
                write(out/'progress.json', dict(step=step, total=steps, elapsed=time.monotonic()-started, losses=losses,
                                                time=time.time(), pid=os.getpid()))
            if step % job['eval_every'] == 0:
                write(out/('eval_%07d.json' % step), evaluate(step))
                evaluations += 1
    finally:
        env.close()
    checkpoint = out/'final_checkpoint.pt'
    torch.save({name: net.state_dict() for name, net in masked.items()}, str(checkpoint))
    write(out/'COMPLETED.json', dict(job=job, steps=steps, finite=finite(), evaluations=evaluations,
        sample_stream_sha256=stream.hexdigest(), model_digest=digest(), checkpoint_sha256=sha(checkpoint),
        mask_refreshes=len(mask_log), mask_first=mask_log[:1], mask_last=mask_log[-1:],
        wall_seconds=time.monotonic()-started, GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        ended=time.time()))


def main():
    job = read(sys.argv[1])
    out = Path(job['output'])
    try:
        run(job, out)
    except BaseException:
        write(out/'FAILED.json', dict(time=time.time(), traceback=traceback.format_exc(), job=job['id']))
        raise


if __name__ == '__main__':
    main()
