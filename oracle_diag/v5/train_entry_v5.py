"""v5 GPU entry; v4 data routing, RNG and original train.py remain unchanged."""
from support_v5 import *
ALL_METHODS=A_METHODS+('ORACLE-W1',)
SINGLE_SOURCE=()
DRAW_COUNTS={m:(64,64) for m in ALL_METHODS}
from task_env import call_shifted_env, call_target_dataset, patch_normalization, paths
from methods_v5 import build, IndexedReplayBuffer, SourceBackup
from data_v4 import verify_source, single_source_dataset, verify_single_buffer, FIELDS
from instrument_v4 import install_loss_trace
import argparse
import ctypes
import importlib
import inspect
import platform
import random
import runpy
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=ALL_METHODS, required=True)
    parser.add_argument('--seed', type=int, choices=range(5), required=True)
    parser.add_argument('--steps', type=int, choices=(5000,500000), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    kind = 'smoke' if args.steps == 5000 else 'formal'
    if args.method == 'ORACLE-W1':
        assert kind == 'smoke' and args.seed == 0, 'Reuse check only'
    check_approval(kind)
    out = args.out.resolve()
    assert OUT in out.parents
    fresh(out)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training requires CUDA; no silent CPU fallback')
    for key in list(os.environ):
        if key.startswith('ODRL_'):
            del os.environ[key]
    coefficient = 1.0
    single = False
    os.environ.update(ODRL_TARGET_FRACTION='1', ODRL_SOURCE_BUDGET=str(coefficient), ODRL_BELLMAN_ETA='1.0')
    if single:
        os.environ['ODRL_TARGET_ONLY'] = '1'
    assert 'ODRL_PAPER_STYLE_SOURCE_TARGET' not in os.environ
    receipt = oracle_receipt()
    oracle = read(receipt)
    provider = SourceBackup('cpu', receipt) if args.method == 'ORACLE-ONLY' else None
    import algo.utils as utils
    utils.ReplayBuffer = IndexedReplayBuffer
    loaded = dict(target_dataset_calls=0, source_route_calls=0)

    def dataset(task, shift_level='medium', quality='medium'):
        assert task == 'walker2d-morph' and str(shift_level) == quality == 'medium'
        if args.method in SINGLE_SOURCE:
            frame = inspect.currentframe().f_back
            assert frame.f_globals.get('__file__') == str(ROOT/'train.py')
            src = frame.f_locals['src_replay_buffer']
            verify_source(src, receipt)
            loaded['source_route_calls'] += 1
            return single_source_dataset(src, args.method, provider)
        loaded['target_dataset_calls'] += 1
        return call_target_dataset(task, shift_level, quality)

    importlib.import_module('dataset.call_dataset').call_tar_dataset = dataset
    shifted_instances = []

    def shifted(config):
        env = call_shifted_env(config)
        shifted_instances.append(env)
        return env

    importlib.import_module('envs.mujoco.call_mujoco_env').call_mujoco_env = shifted
    constants = patch_normalization('walker2d-morph')
    info = importlib.import_module('envs.infos')
    normal = info.get_normalized_score
    evaluations = []

    def normalization(score, name):
        frame = inspect.currentframe().f_back
        assert frame.f_globals.get('__file__') == str(ROOT/'train.py')
        env = frame.f_locals['tar_eval_env']
        assert env is shifted_instances[0] and env._max_episode_steps == 1000
        assert score == frame.f_locals['tar_eval_return']
        value = normal(score, name)
        assert abs(value - (score-constants['min'])/(constants['max']-constants['min'])*100) < 1e-10
        evaluations.append(dict(step=int(frame.f_locals['t'])+1, raw_return=float(score), normalized_score=float(value),
            env_class=type(env.unwrapped).__module__, xml_sha256=sha(paths('walker2d-morph')[0]),
            confirmed_target_env_object=True, lookup_key=name, time_limit=1000))
        write(out/'evaluation_verified.json',dict(normalization=constants,events=evaluations),replace=True)
        return value

    info.get_normalized_score = normalization

    def factory(policy, config, mode, device):
        assert policy == 'IQL' and mode == 3 and str(device).startswith('cuda')
        assert config['batch_size'] == 128 and config['source_budget'] == coefficient and config['bellman_eta'] == 1.
        assert config['target_only'] == single
        if kind == 'smoke':
            config['eval_freq'] = 5000
        assert config['eval_freq'] == (5000 if kind == 'smoke' else 10000)
        model = build(args.method, config, device, receipt)
        trace = install_loss_trace(model)
        start = time.time()
        checked = False
        training_seconds = 0.
        draws = DRAW_COUNTS[args.method]
        write(out/'effective_config.json',dict(config=config,method=args.method,seed=args.seed,
            total_batch=sum(draws),source_draws=draws[0],target_draws=draws[1],standard_direction=True,
            normalization=constants,oracle_receipt=str(receipt),oracle_receipt_sha256=sha(receipt),
            protocol_sha256=sha(OUT/'protocol.json'), purpose=kind,
            learner_target_slot='source' if args.method in SINGLE_SOURCE else 'target',
            tensorboard_target_score_tag='test/target_normalized_score',
            runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))))
        write(out/'initialization.json',dict(seed=args.seed,sha256=model_hash(model)))
        base = model.train

        def instrument(src, tar, batch_size=128, writer=None):
            nonlocal checked, training_seconds
            if not checked:
                assert batch_size == 128
                verify_source(src, receipt)
                if args.method in SINGLE_SOURCE:
                    details = verify_single_buffer(src,tar,args.method,provider)
                    assert loaded == dict(target_dataset_calls=0,source_route_calls=1)
                else:
                    expected = call_target_dataset('walker2d-morph')
                    assert tar.size == 5000 and loaded == dict(target_dataset_calls=1,source_route_calls=0)
                    for attr, key in FIELDS:
                        val = getattr(tar,attr).reshape(expected[key].shape)
                        assert np.array_equal(val,1.-expected[key] if key == 'terminals' else expected[key]), key
                    details = dict(all_rows_equal=True,row_count=5000,target_dataset_used=True,
                                   empirical_target_normalization_used=False)
                write(out/'data_verified.json',dict(source_n=src.size,training_buffer_n=tar.size,
                    all_source_rows_aligned=True,source_dataset_sha256=oracle['source_dataset_sha256'],
                    target_dataset_sha256=None if args.method in SINGLE_SOURCE else sha(paths('walker2d-morph')[1]),
                    routed_data=details,loader_calls=loaded))
                np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed)
                checked = True
            before = (src.draw_count,tar.draw_count)
            tick = time.perf_counter()
            base(src,tar,batch_size,writer)
            training_seconds += time.perf_counter()-tick
            step = model.total_it
            slots = (src.draw_count-before[0],tar.draw_count-before[1])
            actual = (slots[1],0) if args.method in SINGLE_SOURCE else slots
            assert actual == draws, (actual,draws)
            if step == 1:
                source_ids = tar.last_indices if args.method in SINGLE_SOURCE else getattr(src,'last_indices',np.array([],int))
                target_ids = np.array([],int) if args.method in SINGLE_SOURCE else tar.last_indices
                write(out/'first_sample.json',dict(source_ids=source_ids.tolist(),target_ids=target_ids.tolist(),
                    actual_source_draws=actual[0],actual_target_draws=actual[1]))
            if step == 10:
                assert len(trace) == 30
                write(out/'early_losses.json',dict(events=trace,steps=10,device='cuda',historical_v3_trace=False))
            if step % 1000 == 0 or step == args.steps:
                torch.cuda.synchronize()
                assert all(torch.isfinite(v).all() for net in (model.policy,model.v_func,model.q_funcs) for v in net.parameters())
                cumulative = (tar.draw_count,0) if args.method in SINGLE_SOURCE else (src.draw_count,tar.draw_count)
                progress = dict(step=step,elapsed=time.time()-start,pid=os.getpid(),training_seconds=training_seconds,
                    steps_per_second=step/training_seconds,
                    source_draws=cumulative[0],target_draws=cumulative[1],physical_buffer_draws=[src.draw_count,tar.draw_count],
                    q_calls=model._q_calls,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated())
                write(out/'progress.json',progress,replace=True)
                with open(out/'throughput.jsonl','a',encoding='utf-8') as f:f.write(json.dumps(progress)+'\n')
            if step % 100000 == 0 or step == args.steps:
                (out/'values').mkdir(exist_ok=True)
                torch.save(model.v_func.state_dict(),winpath(out/'values'/('V_%d.pt'%step)))

        model.train = instrument
        return model

    importlib.import_module('algo.call_algo').call_algo = factory
    short = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(out),short,len(short)):
        raise RuntimeError('Cannot obtain Windows short output path')
    params = dict(batch_size=128,source_budget=coefficient,source_loss_weight=1.,bellman_eta=1.,target_only=single)
    sys.argv = ['train.py','--policy','IQL','--env','walker2d-morph','--shift_level','medium',
        '--srctype','medium','--tartype','medium','--mode','3','--seed',str(args.seed),
        '--max_step',str(args.steps),'--dir',short.value,'--save-model','--params',json.dumps(params)]
    runpy.run_path(str(ROOT/'train.py'),run_name='__main__')
    assert evaluations and evaluations[-1]['step'] == args.steps
    write(out/'completed.json',dict(method=args.method,seed=args.seed,steps=args.steps,task='walker2d-morph',ended=time.time()))


if __name__ == '__main__':
    main()
