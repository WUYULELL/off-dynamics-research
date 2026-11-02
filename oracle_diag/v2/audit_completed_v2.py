"""CPU-only independent audit of completed v2 artifacts; never trains or overwrites a run."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse, datetime, hashlib, math, statistics, traceback
from pathlib import Path
import numpy as np
import h5py
import torch
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from support import OUT, HERE, ROOT, PREV, read, write, fresh


def extended(path):
    return '\\\\?\\' + str(Path(path).resolve())


def digest(path):
    h = hashlib.sha256()
    with open(extended(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def close(a, b, label):
    assert math.isclose(float(a), float(b), rel_tol=2e-6, abs_tol=1e-10), (label, a, b)


def verify_stats(values, saved, label):
    values = np.asarray(values)
    assert values.size and np.isfinite(values).all(), label
    actual = dict(n=values.size, mean=values.mean(), median=np.median(values),
                  p95=np.percentile(values, 95), max=values.max())
    for name, value in actual.items():
        close(value, saved[name], label + '/' + name)


def events(path):
    """Read raw event protobufs, not report_v2.score/EventAccumulator."""
    scores = {}
    for event in LegacyEventFileLoader(extended(path)).Load():
        for value in event.summary.value:
            if value.tag.replace(' ', '_') != 'test/target_normalized_score':
                continue
            assert value.WhichOneof('value') == 'simple_value'
            assert event.step not in scores, ('duplicate target evaluation', str(path), event.step)
            assert math.isfinite(value.simple_value)
            scores[event.step] = float(value.simple_value)
    assert sorted(scores) == list(range(10000, 500001, 10000)), str(path)
    return scores


def finite(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value: finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def evaluate_v(checkpoint, states):
    """Explicit three-layer saved V evaluation, independent of algorithm wrappers."""
    weights = torch.load(extended(checkpoint), map_location='cpu')
    x = torch.as_tensor(states, dtype=torch.float32)
    with torch.no_grad():
        for i in [0, 2, 4]:
            x = torch.nn.functional.linear(x, weights['network.network.%d.weight' % i],
                                          weights['network.network.%d.bias' % i])
            if i != 4: x = torch.relu(x)
    return x.ravel()


def bellman(checkpoint, ns, reward, nd):
    return (torch.as_tensor(reward, dtype=torch.float32) + .99 *
            torch.as_tensor(nd, dtype=torch.float32) * evaluate_v(checkpoint, ns)).numpy()


def align(raw_path, oracle):
    with h5py.File(str(raw_path), 'r') as h:
        fields = {'source_state': 'observations', 'source_action': 'actions',
                  'source_next': 'next_observations', 'source_reward': 'rewards'}
        for source, key in fields.items():
            actual = h[key][:].astype(np.float32).reshape(oracle[source].shape)
            assert np.array_equal(actual, oracle[source]), (str(raw_path), key)
        assert np.array_equal(1. - h['terminals'][:].astype(bool).reshape(-1), oracle['source_not_done'])
    assert np.array_equal(oracle['indices'], np.arange(5000))
    for key in oracle.files:
        if oracle[key].dtype.kind in 'fiu': assert np.isfinite(oracle[key]).all(), key


def main(attempt):
    fresh(attempt)
    torch.set_num_threads(2)
    res = Path(read(OUT / 'active_execution.json')['directory'])
    jobs = read(res / 'queue_status.json')['jobs']
    assert len(jobs) == 15 and all(j['state'] == 'completed' and j['returncode'] == 0 for j in jobs)
    assert {(j['seed'], j['method']) for j in jobs} == {(s, m) for s in range(5) for m in ['TARGET', 'CONST', 'BACKUP']}
    snapshots = [res/'pipeline_error.json', res/'status.json', OUT/'resume1.stderr.log',
                 HERE/'REPORT_v2.md', OUT/'report_data.json', OUT/'active_execution.json']
    preserved = {}
    for index, path in enumerate(snapshots):
        target = attempt / ('before_%d_%s' % (index, path.name))
        target.write_bytes(path.read_bytes())
        preserved[str(path)] = digest(path)
    write(attempt/'recovery_provenance.json', dict(reason='Completed training; final audit failed on Windows MAX_PATH. CPU-only re-audit using extended absolute paths.',
        preserved_sha256=preserved, queue_sha256=digest(res/'queue_status.json'), training_restarted=False,
        code_sha256=digest(__file__), original_execution_directory=str(res)))
    try:
        protocol = read(OUT/'preregistered_protocol.json')
        plan = read(OUT/'training_plan.json')
        manifest = read(res/'resume_manifest.json')
        for group in [protocol['protected_sha256'], plan['code_sha256'], manifest['retained_prior_partial_files']]:
            for path, expected in group.items(): assert digest(path) == expected, ('changed protected file', path)
        assert digest(OUT/'preregistered_protocol.json') == manifest['protocol_sha256']
        assert read(OUT/'entry_smoke_verified.json')['passed'] and read(OUT/'gpu_validation.json')['passed']
        a = read(OUT/'stage_a/completed.json'); b = read(OUT/'stage_b/completed.json')
        c = read(OUT/'stage_c/completed.json'); extra = read(res/'stage_b_new_tasks/completed.json')
        assert a['passed'] and b['completed'] and c['completed'] and extra['completed']
        assert b['selected'] is None and b['passed_candidates'] == []
        assert len(extra['tasks']) == 4
        # All source/oracle rows and loader corrections are checked against original HDF5.
        for env, row in a['environments'].items():
            assert digest(row['source_path']) == row['source_sha256']
            assert digest(row['oracle_path']) == row['oracle_sha256']
            z = np.load(row['oracle_path']); align(row['source_path'], z)
            with h5py.File(row['source_path'], 'r') as h:
                changed = np.max(np.abs(h['observations'][1:].astype(np.float32) - z['source_next'][:-1]), axis=1)
            ids = np.flatnonzero(changed > 0)
            assert ids.tolist() == row['changed_rows']
            np.testing.assert_allclose(changed[ids], row['changed_row_max_abs'], rtol=0, atol=0)
            old = np.load(PREV/'diagnostics'/(env+'_source_oracle.npz'))
            assert np.flatnonzero(old['boundary']).tolist() == ids.tolist()
            error = np.max(np.abs(z['source_next'] - z['source_physical_next']), axis=1)
            verify_stats(error, row['source_replay_error'], env)
            assert error.max() <= .001 and row['final_row_reward_error'] <= .0001
            close(500000*128/len(z['indices']), row['expected_draws_per_row'], env+'/draws')
        # Original-loader audit is deliberately separate from the formal v2 statistics.
        original_runs = read(OUT/'stage_a/training_variability_original.json')['runs']
        for run in original_runs:
            values = events(run['file']); last = [values[k] for k in range(460000, 500001, 10000)]
            tail = [values[k] for k in range(300000, 500001, 10000)]
            assert last == run['last5'] and tail == run['tail21']
            for key, value in [('last5_mean', statistics.mean(last)), ('last5_std_ddof1', statistics.stdev(last)),
                               ('last5_range', max(last)-min(last)), ('tail21_std_ddof1', statistics.stdev(tail))]:
                close(value, run[key], 'A2/'+key)
        hopper = read(OUT/'stage_a/hopper_q_targets_original.json')
        assert digest(hopper['checkpoint']) == hopper['checkpoint_sha256']
        z = np.load(PREV/'diagnostics/hopper-kinematic-legjnt_source_oracle.npz')
        diff = np.abs(bellman(hopper['checkpoint'], z['source_next'], z['source_reward'], z['source_not_done']) -
                      bellman(hopper['checkpoint'], z['oracle_next'], z['oracle_reward'], z['oracle_not_done']))
        verify_stats(diff, hopper['difference'], 'A2/Q target')
        assert int((diff>1e-4).sum()) == hopper['over_1e4']
        # Original Walker B: recompute state and Bellman errors from estimates and saved V.
        source = np.load(OUT/'stage_a/walker2d-kinematic-footjnt_source_oracle_fixed.npz')
        arrays = np.load(OUT/'stage_b/diagnostic_arrays.npz')
        for key, metrics in b['metrics'].items():
            estimate = np.load(OUT/'stage_b'/('estimate_'+key+'.npz'))
            state_error = np.linalg.norm((estimate['next_state']-source['oracle_next'])/source['target_state_std'], axis=1)
            np.testing.assert_allclose(state_error, arrays[key+'_state'], rtol=1e-6, atol=1e-7)
            verify_stats(state_error, metrics['state'], 'B/'+key)
            close((state_error<arrays['C_state']).mean(), metrics['better_than_C_fraction'], 'B/fraction')
            for kind in ['TARGET', 'MIX_FB']:
                checkpoint = read(PREV/'diagnostics'/('walker2d-kinematic-footjnt_'+kind+'_bellman.json'))['checkpoint']
                y = bellman(checkpoint, estimate['next_state'], estimate['reward'], estimate['not_done'])
                truth = bellman(checkpoint, source['oracle_next'], source['oracle_reward'], source['oracle_not_done'])
                error = np.abs(y-truth)
                np.testing.assert_allclose(error, arrays[key+'_'+kind], rtol=0, atol=1e-4)
                verify_stats(error, metrics['bellman'][kind]['error'], 'B/'+key+'/'+kind)
        close(np.quantile(arrays['validation_disagreement'], .9), b['gate_threshold'], 'B/gate threshold')
        assert np.array_equal(arrays['gate'], arrays['disagreement']<=b['gate_threshold'])
        close(arrays['gate'].mean(), b['source_gate_fraction'], 'B/gate fraction')
        # Model train/validation membership is trajectory-disjoint, including reused Hopper.
        split_checks = []
        for path, episode in [(OUT/'stage_b/split_arrays.npz', 'episode_id'),
                              (res/'stage_b_new_tasks/hopper_split.npz', 'episode')]:
            z = np.load(path); train = z['train'].astype(bool)
            left, right = set(z[episode][train].tolist()), set(z[episode][~train].tolist())
            assert not left.intersection(right)
            split_checks.append(dict(path=str(path),train_rows=int(train.sum()),val_rows=int((~train).sum()),
                                     train_trajectories=len(left),val_trajectories=len(right),overlap=0))
        supplemental_gates = {}; metric_arrays_checked = 15
        for task, row in extra['tasks'].items():
            z = np.load(res/'stage_b_new_tasks'/(task+'_errors.npz'))
            passed = []
            for key, metrics in row['metrics'].items():
                verify_stats(z[key+'_state'], metrics['state'], task+'/'+key+'/state')
                for kind in ['TARGET','MIX_FB']:
                    entry = metrics['bellman'][kind]
                    verify_stats(z[key+'_'+kind], entry['error'], task+'/'+key+'/'+kind)
                    verify_stats(z['C_'+kind], entry['C_error'], task+'/baseline/'+kind)
                metric_arrays_checked += 3
                if key != 'C' and all(z[key+'_'+v].mean()<=.8*z['C_'+v].mean() and
                                     np.median(z[key+'_'+v])<=.8*np.median(z['C_'+v]) for v in ['TARGET','MIX_FB']):
                    passed.append(key)
            supplemental_gates[task] = passed
        # C replay arrays and physical shift, preserving original thresholds.
        for task, row in c['datasets'].items():
            z = np.load(OUT/'stage_c'/(task+'_replay.npz'))
            o = np.load(row['oracle_path'])
            kind, shift = task.split('_')
            raw = ROOT/'reproductions/OTDF/datasets'/(kind+'_'+('gravity_0.5' if shift=='gravity' else shift)+'_medium.hdf5')
            assert digest(raw)==row['source_sha256'] and digest(row['xml'])==row['xml_sha256']
            assert digest(row['oracle_path'])==row['oracle_sha256']
            align(raw,o)
            assert row['accepted'] and z['passed'].all() and row['done_disagreements']==0
            assert z['state_error'].max()<=.001 and z['reward_error'].max()<=.0001
            for metric in ['state_error','reward_error']: verify_stats(z[metric],row[metric],task+'/'+metric)
            shift_error=np.linalg.norm((o['source_physical_next']-o['oracle_next'])/o['target_state_std'],axis=1)
            verify_stats(shift_error,row['physical_shift'],task+'/shift')
            close((shift_error>1e-3).mean(),row['shift_over_1e3_fraction'],task+'/shift fraction')
        print('A/B/C and four supplemental tasks verified',flush=True)
        # Formal runs: raw events, model/optimizer files, V snapshots, pairing and configuration.
        runs=[]; model_files=[]; run_paths={}; common_configs={}
        prior_report=read(OUT/'report_data.json')
        for job in jobs:
            folder=Path(job['out']); seed=job['seed']; method=job['method']; run_paths[seed,method]=folder
            marker=read(folder/'completed.json'); assert marker['steps']==500000 and marker['seed']==seed and marker['method']==method
            config=read(folder/'effective_config.json'); cfg=config['config']
            assert cfg['seed']==seed and cfg['batch_size']==128 and cfg['max_step']==500000
            assert cfg['gamma']==.99 and cfg['actor_lr']==cfg['critic_lr']==.0003
            if method=='CONST': assert cfg['const_source_weight']==cfg['source_budget']==.20
            if method=='BACKUP': assert cfg['source_budget']==.20 and cfg['bellman_eta']==1.
            assert digest(config['oracle_path'])==config['oracle_sha256']
            aligned=read(folder/'data_verified.json'); assert aligned['aligned'] and aligned['source_n']==5000 and aligned['target_n']==99932
            event_files=list(folder.rglob('events.out.tfevents*')); assert len(event_files)==1
            values=events(event_files[0]); last=[values[k] for k in range(460000,500001,10000)]
            mean=statistics.mean(last)
            previous=next(x for x in prior_report['scores'] if x['seed']==seed and x['method']==method)
            assert last==previous['values']; close(mean,previous['mean'],'D/report reconciliation')
            files=list(folder.rglob('model_*')); assert len(files)==7
            files+=list((folder/'values').glob('V_*.pt')); assert len(files)==12
            loaded={}
            for path in files:
                weights=torch.load(extended(path),map_location='cpu'); finite(weights); loaded[path.name]=weights
                model_files.append(dict(path=str(path),sha256=digest(path),finite=True))
            assert loaded['model_value'].keys()==loaded['V_500000.pt'].keys()
            for key in loaded['model_value']: assert torch.equal(loaded['model_value'][key],loaded['V_500000.pt'][key])
            runs.append(dict(seed=seed,method=method,last5=last,mean=mean,file=str(event_files[0]),
                             event_sha256=digest(event_files[0]),evaluation_count=len(values),steps=sorted(values),
                             scores=[values[k] for k in sorted(values)],wall_hours=(job['ended']-job['started'])/3600))
            print('Verified seed',seed,method,'Last5',mean,flush=True)
        for seed in range(5):
            paths=[run_paths[seed,m] for m in ['TARGET','CONST','BACKUP']]
            assert len({read(p/'initialization.json')['sha256'] for p in paths})==1
            assert len({read(p/'data_verified.json')['target_sha256'] for p in paths})==1
            assert read(paths[1]/'first_sample.json')==read(paths[2]/'first_sample.json')
        score={(x['seed'],x['method']):x['mean'] for x in runs}; paired=[]; summaries={}
        for method,base in [('BACKUP','CONST'),('CONST','TARGET'),('BACKUP','TARGET')]:
            delta=[score[s,method]-score[s,base] for s in range(5)]
            mean=statistics.mean(delta); sd=statistics.stdev(delta); se=sd/math.sqrt(5)
            paired.append(dict(method=method,baseline=base,delta=delta,mean=mean,sd=sd,se=se,
                               positive=sum(x>0 for x in delta),passed=mean>2*se and sum(x>0 for x in delta)>=4,
                               role='primary' if method=='BACKUP' and base=='CONST' else 'secondary_descriptive'))
        for method in ['TARGET','CONST','BACKUP']:
            means=[score[s,method] for s in range(5)]
            summaries[method]=dict(mean=statistics.mean(means),sd=statistics.stdev(means),se=statistics.stdev(means)/math.sqrt(5))
        for previous,current in zip(prior_report['paired'],paired):
            assert previous['delta']==current['delta']
            for key in ['mean','sd','se','positive','passed']: close(previous[key],current[key],'D/paired/'+key)
        assert not torch.cuda.is_initialized()
        validation=dict(passed=True,checked_at=datetime.datetime.now().astimezone().isoformat(),execution_directory=str(res),
            attempt_directory=str(attempt),training_jobs=15,model_and_snapshot_files=len(model_files),
            protected_unchanged=True,protected_files=len(protocol['protected_sha256']),core_files=len(plan['code_sha256']),
            retained_prior_partial_files=len(manifest['retained_prior_partial_files']),pairing_verified=True,
            source_oracle_hdf5_alignment=True,raw_tensorboard_events_verified=True,original_audit_runs=len(original_runs),
            diagnostic_metric_arrays_verified=metric_arrays_checked,trajectory_splits=split_checks,
            supplemental_passed_candidates=supplemental_gates,cuda_initialized=False,
            known_repair='Extended Windows paths for final CPU checkpoint loading; no training or scientific change',
            total_training_hours=sum(x['wall_hours'] for x in runs),
            elapsed_queue_hours=(max(j['ended'] for j in jobs)-min(j['started'] for j in jobs))/3600)
        write(attempt/'audit.json',validation)
        write(attempt/'independent_statistics.json',dict(runs=runs,paired=paired,method_summaries=summaries))
        write(attempt/'model_manifest.json',dict(files=model_files))
        print('AUDIT PASSED',validation['model_and_snapshot_files'],'model/snapshot files',flush=True)
        print(summaries,paired,flush=True)
    except BaseException:
        write(attempt/'failure.json',dict(error=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    main(p.parse_args().attempt.resolve())
