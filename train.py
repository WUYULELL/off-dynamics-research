import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
import yaml
import json # in case the user want to modify the hyperparameters

local_mujoco_path = os.path.expanduser('~/.mujoco210/mujoco210')
if os.path.isdir(local_mujoco_path):
    os.environ.setdefault('MUJOCO_PY_MUJOCO_PATH', local_mujoco_path)
    os.environ['PATH'] = os.path.join(local_mujoco_path, 'bin') + os.pathsep + os.environ.get('PATH', '')

import d4rl # used to make offline environments for source domains
import algo.utils as utils

from pathlib                              import Path
from algo.call_algo                       import call_algo
from dataset.call_dataset                 import call_tar_dataset
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.adroit.call_adroit_env          import call_adroit_env
from envs.antmaze.call_antmaze_env        import call_antmaze_env
from envs.infos                           import get_normalized_score
from tensorboardX                         import SummaryWriter


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="SAC", help='policy to use')
    parser.add_argument("--env", default="halfcheetah-friction")
    parser.add_argument('--srctype', default="medium", help='dataset type used in the source domain') # only useful when source domain is offline
    parser.add_argument('--tartype', default="medium", help='dataset type used in the target domain') # only useful when target domain is offline
    # support dataset type:
    # source domain: all valid datasets from D4RL
    # target domain: random, medium, medium-expert, expert
    parser.add_argument('--shift_level', default=0.1, help='the scale of the dynamics shift. Note that this value varies on different settins')
    parser.add_argument('--mode', default=0, type=int, help='the training mode, there are four types, 0: online-online, 1: offline-online, 2: online-offline, 3: offline-offline')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument('--tar_env_interact_interval', help='interval of interacting with target env', default=10, type=int)
    parser.add_argument('--max_step', default=int(1e6), type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    
    args = parser.parse_args()

    # we support different ways of specifying tasks, e.g., hopper-friction, hopper_friction, hopper_morph_torso_easy, hopper-morph-torso-easy
    if '_' in args.env:
        args.env = args.env.replace('_', '-')

    if 'halfcheetah' in args.env or 'hopper' in args.env or 'walker2d' in args.env or args.env.split('-')[0] == 'ant':
        domain = 'mujoco'
    elif 'pen' in args.env or 'relocate' in args.env or 'door' in args.env or 'hammer' in args.env:
        domain = 'adroit'
    elif 'antmaze' in args.env:
        domain = 'antmaze'
    else:
        raise NotImplementedError
    print(domain)

    call_env = {
        'mujoco': call_mujoco_env,
        'adroit': call_adroit_env,
        'antmaze': call_antmaze_env,
    }

    # determine referenced environment name
    ref_env_name = args.env + '-' + str(args.shift_level)
    
    if domain == 'antmaze':
        src_env_name = args.env
        src_env_name_config = args.env
    elif domain == 'adroit':
        src_env_name = args.env
        src_env_name_config = args.env.split('-')[0]
    else:
        src_env_name = args.env.split('-')[0]
        src_env_name_config = src_env_name
    tar_env_name = args.env

    # make environments
    if args.mode == 1 or args.mode == 3:
        if domain == 'antmaze':
            src_env_name = src_env_name.split('-')[0]
            src_env_name += '-' + args.srctype + '-v0'
        elif domain == 'adroit':
            src_env_name = src_env_name.split('-')[0]
            src_env_name += '-' + args.srctype + '-v0'
        else:
            src_env_name += '-' + args.srctype + '-v2'
        src_env = None
        src_eval_env = gym.make(src_env_name)
        src_eval_env.seed(args.seed)
    else:
        if 'antmaze' in src_env_name:
            src_env_config = {
                'env_name': src_env_name,
                'shift_level': None,
            }
            src_env = call_env[domain](src_env_config)
            src_env.seed(args.seed)
            src_eval_env = call_env[domain](src_env_config)
            src_eval_env.seed(args.seed + 100)
        else:
            src_env_config = {
                'env_name': src_env_name,
                'shift_level': None,
            }
            src_env = call_env[domain](src_env_config)
            src_env.seed(args.seed)
            src_eval_env = copy.deepcopy(src_env)
            src_eval_env.seed(args.seed + 100)

    if args.mode == 2 or args.mode == 3:
        tar_env = None
        tar_env_config = {
            'env_name': tar_env_name,
            'shift_level': args.shift_level,
        }
        tar_eval_env = call_env[domain](tar_env_config)
        tar_eval_env.seed(args.seed + 100)
    else:
        tar_env_config = {
            'env_name': tar_env_name,
            'shift_level': args.shift_level,
        }
        tar_env = call_env[domain](tar_env_config)
        tar_env.seed(args.seed)
        tar_eval_env = call_env[domain](tar_env_config)
        tar_eval_env.seed(args.seed + 100)
    
    if args.mode not in [0,1,2,3]:
        raise NotImplementedError # cannot support other modes
    
    policy_config_name = args.policy.lower()
    if policy_config_name in ['iql_dyn', 'iql_dyn_unc_fb', 'iql_dyn_unc_light', 'iql_dyn_unc_warmup']:
        policy_config_name = 'iql_dyn'
    elif policy_config_name == 'iql_bellman_ab_v5':
        policy_config_name = 'iql_bellman_ab'
    elif policy_config_name in ['iql_bellman_fb', 'iql_bellman_ab']:
        policy_config_name = policy_config_name

    # load pre-defined hyperparameter config for training
    with open(f"{str(Path(__file__).parent.absolute())}/config/{domain}/{policy_config_name}/{src_env_name_config}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.params is not None:
        override_params = json.loads(args.params)
        config.update(override_params)
        print('The following parameters are updated to:', args.params)

    if os.environ.get('ODRL_TARGET_ONLY', '0') == '1':
        config['target_only'] = True
        print('The following parameters are updated by environment: {"target_only": true}')

    policy_upper = args.policy.upper()
    if policy_upper == 'IQL_DYN':
        config['reweight_mode'] = 'dyn'
        print('The following IQL_DYN parameters are updated by policy: {"reweight_mode": "dyn"}')
    elif policy_upper == 'IQL_DYN_UNC_FB':
        config['reweight_mode'] = 'dyn_unc'
        print('The following IQL_DYN_UNC_FB parameters are updated by policy: {"reweight_mode": "dyn_unc"}')
    elif policy_upper == 'IQL_DYN_UNC_LIGHT':
        config.update({
            'reweight_mode': 'dyn_unc',
            'dyn_alpha': 1.0,
            'unc_alpha': 0.1,
            'unc_factor_min': 0.7,
            'source_budget': 0.33,
            'min_weight': 0.10,
            'max_weight': 1.0,
        })
        print('The following IQL_DYN_UNC_LIGHT parameters are updated by policy:', {
            'reweight_mode': 'dyn_unc',
            'dyn_alpha': config['dyn_alpha'],
            'unc_alpha': config['unc_alpha'],
            'unc_factor_min': config['unc_factor_min'],
            'source_budget': config['source_budget'],
            'min_weight': config['min_weight'],
            'max_weight': config['max_weight'],
        })
    elif policy_upper == 'IQL_DYN_UNC_WARMUP':
        config.update({
            'reweight_mode': 'dyn_unc',
            'dyn_alpha': 1.0,
            'unc_alpha': 0.1,
            'unc_factor_min': 0.7,
            'unc_start_step': 100000,
            'source_budget': 0.33,
            'min_weight': 0.10,
            'max_weight': 1.0,
        })
        print('The following IQL_DYN_UNC_WARMUP parameters are updated by policy:', {
            'reweight_mode': 'dyn_unc',
            'dyn_alpha': config['dyn_alpha'],
            'unc_alpha': config['unc_alpha'],
            'unc_factor_min': config['unc_factor_min'],
            'unc_start_step': config['unc_start_step'],
            'source_budget': config['source_budget'],
            'min_weight': config['min_weight'],
            'max_weight': config['max_weight'],
        })
    elif policy_upper == 'IQL_BELLMAN_FB':
        config['source_budget'] = config.get('source_budget', 0.33)
        print('The following IQL_BELLMAN_FB parameters are updated by policy:', {
            'source_budget': config['source_budget'],
            'bellman_eta': config.get('bellman_eta', 1.0),
            'normalize_bellman': config.get('normalize_bellman', True),
        })
    elif policy_upper == 'IQL_BELLMAN_AB':
        config['budget_mode'] = config.get('budget_mode', 'adaptive')
        print('The following IQL_BELLMAN_AB parameters are updated by policy:', {
            'budget_mode': config['budget_mode'],
            'budget_min': config.get('budget_min', 0.10),
            'budget_max': config.get('budget_max', 0.33),
            'bellman_alpha': config.get('bellman_alpha', 1.0),
            'adaptive_budget_update_interval': config.get('adaptive_budget_update_interval', 10),
        })
    elif policy_upper == 'IQL_BELLMAN_AB_V5':
        config.update({
            'budget_mode': 'adaptive',
            'budget_anchor': 0.20,
            'budget_min': 0.10,
            'budget_max': 0.25,
            'budget_ratio_center': 6.5,
            'budget_response_scale': 2.0,
            'diagnostic_log_interval': 5000,
        })
        print('The following IQL_BELLMAN_AB_V5 parameters are updated by policy:', {
            'budget_mode': config['budget_mode'],
            'budget_anchor': config['budget_anchor'],
            'budget_min': config['budget_min'],
            'budget_max': config['budget_max'],
            'budget_ratio_center': config['budget_ratio_center'],
            'budget_response_scale': config['budget_response_scale'],
            'diagnostic_log_interval': config['diagnostic_log_interval'],
        })

    if policy_upper == 'IQL_UNC':
        unc_env_overrides = {
            'ODRL_UNC_ALPHA': ('unc_alpha', float),
            'ODRL_UNC_MIN_WEIGHT': ('unc_min_weight', float),
            'ODRL_SOURCE_LOSS_WEIGHT': ('source_loss_weight', float),
            'ODRL_CONST_SOURCE_WEIGHT': ('const_source_weight', float),
        }
        applied_unc_overrides = {}
        for env_name, (config_name, converter) in unc_env_overrides.items():
            env_value = os.environ.get(env_name)
            if env_value is not None:
                config[config_name] = converter(env_value)
                applied_unc_overrides[config_name] = config[config_name]
        if applied_unc_overrides:
            print('The following IQL_UNC parameters are updated by environment:', applied_unc_overrides)

    if policy_upper in ['IQL_DYN', 'IQL_DYN_UNC_FB', 'IQL_DYN_UNC_LIGHT', 'IQL_DYN_UNC_WARMUP']:
        dyn_env_overrides = {
            'ODRL_DYN_ALPHA': ('dyn_alpha', float),
            'ODRL_UNC_ALPHA': ('unc_alpha', float),
            'ODRL_UNC_FACTOR_MIN': ('unc_factor_min', float),
            'ODRL_UNC_START_STEP': ('unc_start_step', int),
            'ODRL_SOURCE_BUDGET': ('source_budget', float),
            'ODRL_MIN_WEIGHT': ('min_weight', float),
            'ODRL_MAX_WEIGHT': ('max_weight', float),
            'ODRL_KNN_K': ('knn_k', int),
            'ODRL_TARGET_BANK_SIZE': ('target_bank_size', int),
            'ODRL_KNN_CHUNK_SIZE': ('knn_chunk_size', int),
            'ODRL_SOURCE_LOSS_WEIGHT': ('source_loss_weight', float),
        }
        applied_dyn_overrides = {}
        for env_name, (config_name, converter) in dyn_env_overrides.items():
            env_value = os.environ.get(env_name)
            if env_value is not None:
                config[config_name] = converter(env_value)
                applied_dyn_overrides[config_name] = config[config_name]
        if applied_dyn_overrides:
            print('The following IQL_DYN parameters are updated by environment:', applied_dyn_overrides)

    if policy_upper == 'IQL_BELLMAN_FB':
        bellman_env_overrides = {
            'ODRL_BELLMAN_ETA': ('bellman_eta', float),
            'ODRL_SOURCE_BUDGET': ('source_budget', float),
            'ODRL_MIN_WEIGHT': ('min_weight', float),
            'ODRL_MAX_WEIGHT': ('max_weight', float),
            'ODRL_KNN_K': ('knn_k', int),
            'ODRL_TARGET_BANK_SIZE': ('target_bank_size', int),
            'ODRL_KNN_CHUNK_SIZE': ('knn_chunk_size', int),
            'ODRL_SOURCE_LOSS_WEIGHT': ('source_loss_weight', float),
        }
        applied_bellman_overrides = {}
        for env_name, (config_name, converter) in bellman_env_overrides.items():
            env_value = os.environ.get(env_name)
            if env_value is not None:
                config[config_name] = converter(env_value)
                applied_bellman_overrides[config_name] = config[config_name]
        normalize_bellman_value = os.environ.get('ODRL_NORMALIZE_BELLMAN')
        if normalize_bellman_value is not None:
            config['normalize_bellman'] = normalize_bellman_value.lower() in ['1', 'true', 'yes']
            applied_bellman_overrides['normalize_bellman'] = config['normalize_bellman']
        if applied_bellman_overrides:
            print('The following IQL_BELLMAN_FB parameters are updated by environment:', applied_bellman_overrides)

    if policy_upper in ['IQL_BELLMAN_AB', 'IQL_BELLMAN_AB_V5']:
        bellman_ab_env_overrides = {
            'ODRL_BELLMAN_ALPHA': ('bellman_alpha', float),
            'ODRL_BELLMAN_ETA': ('bellman_alpha', float),
            'ODRL_BUDGET_MODE': ('budget_mode', str),
            'ODRL_FIXED_SOURCE_BUDGET': ('fixed_source_budget', float),
            'ODRL_BUDGET_MIN': ('budget_min', float),
            'ODRL_BUDGET_MAX': ('budget_max', float),
            'ODRL_BUDGET_TEMP': ('budget_temp', float),
            'ODRL_BUDGET_RATIO_CENTER': ('budget_ratio_center', float),
            'ODRL_BUDGET_ANCHOR': ('budget_anchor', float),
            'ODRL_BUDGET_RESPONSE_SCALE': ('budget_response_scale', float),
            'ODRL_BUDGET_EMA_DECAY': ('budget_ema_decay', float),
            'ODRL_ADAPTIVE_BUDGET_UPDATE_INTERVAL': ('adaptive_budget_update_interval', int),
            'ODRL_DIAGNOSTIC_LOG_INTERVAL': ('diagnostic_log_interval', int),
            'ODRL_MIN_WEIGHT': ('min_weight', float),
            'ODRL_MAX_WEIGHT': ('max_weight', float),
            'ODRL_KNN_K': ('knn_k', int),
            'ODRL_TARGET_BANK_SIZE': ('target_bank_size', int),
            'ODRL_KNN_CHUNK_SIZE': ('knn_chunk_size', int),
            'ODRL_SOURCE_LOSS_WEIGHT': ('source_loss_weight', float),
        }
        applied_bellman_ab_overrides = {}
        for env_name, (config_name, converter) in bellman_ab_env_overrides.items():
            env_value = os.environ.get(env_name)
            if env_value is not None:
                config[config_name] = converter(env_value)
                applied_bellman_ab_overrides[config_name] = config[config_name]
        normalize_bellman_value = os.environ.get('ODRL_NORMALIZE_BELLMAN')
        if normalize_bellman_value is not None:
            config['normalize_bellman_mismatch'] = normalize_bellman_value.lower() in ['1', 'true', 'yes']
            applied_bellman_ab_overrides['normalize_bellman_mismatch'] = config['normalize_bellman_mismatch']
        use_budget_ema_value = os.environ.get('ODRL_USE_BUDGET_EMA')
        if use_budget_ema_value is not None:
            config['use_budget_ema'] = use_budget_ema_value.lower() in ['1', 'true', 'yes']
            applied_bellman_ab_overrides['use_budget_ema'] = config['use_budget_ema']
        exclude_self_value = os.environ.get('ODRL_EXCLUDE_SELF_IN_TARGET_KNN')
        if exclude_self_value is not None:
            config['exclude_self_in_target_knn'] = exclude_self_value.lower() in ['1', 'true', 'yes']
            applied_bellman_ab_overrides['exclude_self_in_target_knn'] = config['exclude_self_in_target_knn']
        if applied_bellman_ab_overrides:
            print('The following {} parameters are updated by environment:'.format(policy_upper), applied_bellman_ab_overrides)

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    # log path, we use logging with tensorboard
    if args.mode == 1:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-srcdatatype-' + args.srctype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    elif args.mode == 2:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    elif args.mode == 3:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    else:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-' + str(args.shift_level) + '/r' + str(args.seed)
    outdir = os.path.abspath(os.path.normpath(outdir))
    tbdir = os.path.join(outdir, 'tb')
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(tbdir, exist_ok=True)
    writer = SummaryWriter(tbdir)
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    # seed all
    src_env.action_space.seed(args.seed) if src_env is not None else None
    tar_env.action_space.seed(args.seed) if tar_env is not None else None
    src_eval_env.action_space.seed(args.seed)
    tar_eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # get necessary information from both domains
    state_dim = src_eval_env.observation_space.shape[0]
    action_dim = src_eval_env.action_space.shape[0] 
    max_action = float(src_eval_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine shift_level
    if domain == 'mujoco':
        if args.shift_level in ['easy', 'medium', 'hard']:
            shift_level = args.shift_level
        else:
            shift_level = float(args.shift_level)
    else:
        shift_level = args.shift_level

    config.update({
        'env_name': args.env,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'tar_env_interact_interval': int(args.tar_env_interact_interval),
        'max_step': int(args.max_step),
        'shift_level': shift_level,
        'seed': int(args.seed),
    })

    policy = call_algo(args.policy, config, args.mode, device)
    
    ## write logs to record training parameters
    with open(outdir + 'log.txt','w') as f:
        f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))

    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    paper_style_source_target = os.environ.get('ODRL_PAPER_STYLE_SOURCE_TARGET', '0') == '1'
    target_fraction = float(os.environ.get('ODRL_TARGET_FRACTION', '1.0'))

    # in case that the domain is offline, we directly load its offline data
    if args.mode == 1 or args.mode == 3:
        src_replay_buffer.convert_D4RL(d4rl.qlearning_dataset(src_eval_env))
        if paper_style_source_target and args.mode == 3 and target_fraction < 1.0:
            src_replay_buffer.subsample(target_fraction, seed=args.seed)
            print(f'Paper-style target replay buffer subsampled to {target_fraction:.3f}: {src_replay_buffer.size} transitions')
        if 'antmaze' in args.env:
            src_replay_buffer.reward -= 1.0
    
    if args.mode == 2 or args.mode == 3:
        tar_dataset = call_tar_dataset(tar_env_name, shift_level, args.tartype)
        tar_replay_buffer.convert_D4RL(tar_dataset)
        if (not paper_style_source_target) and target_fraction < 1.0:
            tar_replay_buffer.subsample(target_fraction, seed=args.seed)
            print(f'Target replay buffer subsampled to {target_fraction:.3f}: {tar_replay_buffer.size} transitions')
        if 'antmaze' in args.env:
            tar_replay_buffer.reward -= 1.0

    if paper_style_source_target and args.mode == 3:
        src_replay_buffer, tar_replay_buffer = tar_replay_buffer, src_replay_buffer
        src_eval_env, tar_eval_env = tar_eval_env, src_eval_env
        print('Paper-style source/target enabled: source=shifted ODRL data, target=D4RL data/eval env')
        print(f'Paper-style source buffer size: {src_replay_buffer.size}')
        print(f'Paper-style target buffer size: {tar_replay_buffer.size}')

    eval_cnt = 0
    
    eval_src_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
    eval_tar_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
    eval_cnt += 1

    if args.mode == 0:
        # online-online learning

        src_state, src_done = src_env.reset(), False
        tar_state, tar_done = tar_env.reset(), False
        src_episode_reward, src_episode_timesteps, src_episode_num = 0, 0, 0
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            src_episode_timesteps += 1

            # select action randomly or according to policy, if the policy is deterministic, add exploration noise akin to TD3 implementation
            src_action = (
                policy.select_action(np.array(src_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
            ).clip(-max_action, max_action)

            src_next_state, src_reward, src_done, _ = src_env.step(src_action) 
            src_done_bool = float(src_done) if src_episode_timesteps < src_env._max_episode_steps else 0

            if 'antmaze' in args.env:
                src_reward -= 1.0

            src_replay_buffer.add(src_state, src_action, src_next_state, src_reward, src_done_bool)

            src_state = src_next_state
            src_episode_reward += src_reward
            
            # interaction with tar env
            if t % config['tar_env_interact_interval'] == 0:
                tar_episode_timesteps += 1
                tar_action = policy.select_action(np.array(tar_state), test=False)

                tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)
                tar_done_bool = float(tar_done) if tar_episode_timesteps < src_env._max_episode_steps else 0

                if 'antmaze' in args.env:
                    tar_reward -= 1.0

                tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

                tar_state = tar_next_state
                tar_episode_reward += tar_reward

            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
            
            if src_done: 
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, src_episode_num+1, src_episode_timesteps, src_episode_reward))
                writer.add_scalar('train/source return', src_episode_reward, global_step = t+1)

                src_state, src_done = src_env.reset(), False
                src_episode_reward = 0
                src_episode_timesteps = 0
                src_episode_num += 1
            
            if tar_done:
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, tar_episode_num+1, tar_episode_timesteps, tar_episode_reward))
                writer.add_scalar('train/target return', tar_episode_reward, global_step = t+1)
                # record normalized score
                train_normalized_score = get_normalized_score(tar_episode_reward, ref_env_name)
                writer.add_scalar('train/target normalized score', train_normalized_score, global_step = t+1)

                tar_state, tar_done = tar_env.reset(), False
                tar_episode_reward = 0
                tar_episode_timesteps = 0
                tar_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                # record normalized score
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    elif args.mode == 1:
        # offline-online learning
        tar_state, tar_done = tar_env.reset(), False
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            
            # interaction with tar env
            if t % config['tar_env_interact_interval'] == 0:
                tar_episode_timesteps += 1
                tar_action = policy.select_action(np.array(tar_state), test=False)

                tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)
                tar_done_bool = float(tar_done) if tar_episode_timesteps < src_eval_env._max_episode_steps else 0

                if 'antmaze' in args.env:
                    tar_reward -= 1.0

                tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

                tar_state = tar_next_state
                tar_episode_reward += tar_reward

            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
            
            if tar_done:
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, tar_episode_num+1, tar_episode_timesteps, tar_episode_reward))
                writer.add_scalar('train/target return', tar_episode_reward, global_step = t+1)
                train_normalized_score = get_normalized_score(tar_episode_reward, ref_env_name)
                writer.add_scalar('train/target normalized score', train_normalized_score, global_step = t+1)

                tar_state, tar_done = tar_env.reset(), False
                tar_episode_reward = 0
                tar_episode_timesteps = 0
                tar_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    elif args.mode == 2:
        # online-offline learning
        src_state, src_done = src_env.reset(), False
        src_episode_reward, src_episode_timesteps, src_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            src_episode_timesteps += 1

            # select action randomly or according to policy, if the policy is deterministic, add exploration noise akin to TD3 implementation
            src_action = (
                policy.select_action(np.array(src_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
            ).clip(-max_action, max_action)

            src_next_state, src_reward, src_done, _ = src_env.step(src_action) 
            src_done_bool = float(src_done) if src_episode_timesteps < src_env._max_episode_steps else 0

            if 'antmaze' in args.env:
                src_reward -= 1.0

            src_replay_buffer.add(src_state, src_action, src_next_state, src_reward, src_done_bool)

            src_state = src_next_state
            src_episode_reward += src_reward

            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
            
            if src_done: 
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, src_episode_num+1, src_episode_timesteps, src_episode_reward))
                writer.add_scalar('train/source return', src_episode_reward, global_step = t+1)

                src_state, src_done = src_env.reset(), False
                src_episode_reward = 0
                src_episode_timesteps = 0
                src_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    else:
        # offline-offline learning
        for t in range(int(config['max_step'])):
            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)
                
                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    writer.close()
