# -*- coding: utf-8 -*-
"""Full-state restoration and two-stream recovery; no scientific changes."""
import numpy as np
import torch

def restore_training(model, checkpoint):
    saved = torch.load(str(checkpoint), map_location=model.device)
    assert saved['config'] == model.config
    for attr, key in (('policy', 'policy'), ('v_func', 'V'), ('q_funcs', 'Q'), ('target_q_funcs', 'target_Q')):
        getattr(model, attr).load_state_dict(saved[key])
    for attr, key in (('v_optimizer', 'V_optimizer'), ('q_optimizer', 'Q_optimizer'), ('policy_optimizer', 'actor_optimizer')):
        getattr(model, attr).load_state_dict(saved[key])
    model.policy_lr_schedule.load_state_dict(saved['actor_scheduler'])
    model.total_it = saved['step']
    assert model.policy_lr_schedule.T_max == model.config['max_step'] == 500000
    assert model.policy_lr_schedule.last_epoch == model.total_it
    assert model.policy_optimizer.param_groups[0]['lr'] == saved['actor_optimizer']['param_groups'][0]['lr']
    assert all(float(v['step']) == model.total_it for opt in
               (model.v_optimizer, model.q_optimizer, model.policy_optimizer) for v in opt.state.values())
    return saved

def advance_sampler(sampler, updates):
    initial_draws = sampler.draws
    for begin in range(0, updates, 4096):
        n = min(4096, updates - begin)
        ix = sampler.ids[sampler.rng.randint(len(sampler.ids), size=n * 64)]
        if sampler.first is None:
            sampler.first = ix[:64].tolist()
        sampler.hash.update(ix.astype('<i8').tobytes())
        sampler.draws += len(ix)
    assert sampler.draws == initial_draws + updates * 64
    return sampler

def tensor_equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tensor_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(tensor_equal(x, y) for x, y in zip(a, b))
    return a == b
