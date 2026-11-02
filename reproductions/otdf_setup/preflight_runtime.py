"""Explicit one-shot CUDA/model/target-simulator check; no optimizer updates."""
import json
from pathlib import Path
from export_ot_inputs import REPO, configure_environment, prepare_official_import_path


def main():
    configure_environment(Path('C:/Users/TXin/.d4rl/datasets'),
                          Path('C:/Users/TXin/.mujoco210/mujoco210'))
    prepare_official_import_path()
    import numpy as np
    import torch
    import yaml
    from envs.common import call_env
    from algo.OTDF import OTDF
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; do not silently run the experiment on CPU')
    config = yaml.safe_load((REPO / 'config/otdf/walker2d_morph.yaml').read_text(encoding='utf-8'))
    env = call_env(config['tar_env_config'])
    try:
        env.seed(1)
        state = env.reset()
        config.update(state_dim=env.observation_space.shape[0],
                      action_dim=env.action_space.shape[0],
                      max_action=float(env.action_space.high[0]),
                      weight=True, proportion=0.8, noreg=False, reg_weight=0.5)
        torch.manual_seed(1)
        model = OTDF(config, torch.device('cuda'))
        rewards = []
        for _ in range(10):
            action = model.select_action(np.asarray(state))
            state, reward, done, _ = env.step(action)
            if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(reward):
                raise RuntimeError('Non-finite target simulation or CUDA policy output')
            rewards.append(float(reward))
            if done:
                state = env.reset()
        states = torch.as_tensor(np.repeat(np.asarray(state)[None, :], 128, axis=0),
                                 dtype=torch.float32, device='cuda')
        actions = torch.zeros((128, config['action_dim']), device='cuda')
        reconstruction, mean, std = model.vae_policy(states, actions)
        loss = (reconstruction - actions).square().mean() + mean.square().mean() + std.square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.vae_policy.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
            raise RuntimeError('Non-finite or absent CVAE CUDA gradients')
        torch.cuda.synchronize()
        print(json.dumps(dict(status='passed', scope='runtime only; no optimizer steps or checkpoint',
                              gpu=torch.cuda.get_device_name(0), policy_device=str(next(model.policy.parameters()).device),
                              state_dim=config['state_dim'], action_dim=config['action_dim'],
                              target_steps=10, rewards=rewards, cvae_backward_finite=True,
                              torch_peak_allocated_bytes=torch.cuda.max_memory_allocated()), indent=2), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
