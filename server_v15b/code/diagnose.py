"""V15b: read-only diagnosis of the V15 B0 self-check. No simulation, no training, no GPU.

Reads the per-step arrays V15 saved for each target segment (B0_trajectory*.npz),
the V15 B0_RESULT.json / PLAN.json, and the target dataset. Reports, per segment,
the first step at which the tracking residual exceeds several fixed levels, how
the residual grows along the segment, and whether those steps coincide with
clipped recorded velocities, actions on the bound, or recorded actions outside
[-1, 1]. Writes only under $HOME/v15b_b0diag/outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SHIM = os.environ.get('V15B_TEST_SHIM')       # synthetic local test only; never set on the server
if SHIM:
    sys.path.insert(0, SHIM)

import numpy as np

HERE = Path(__file__).resolve().parent
V15 = Path.home()/'v15_screen'
PROJECT = Path.home()/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
NQ = {'hopper-morph': 6, 'walker2d-morph': 9}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(str(tmp), str(path))


def first_above(values, level):
    idx = np.flatnonzero(np.asarray(values) > level)
    return None if not len(idx) else int(idx[0])+1          # 1-based executed step


def pct(a, q):
    return None if not len(a) else float(np.percentile(a, q))


def segment_report(task, seg, npz, data, cfg):
    with np.load(npz, allow_pickle=False) as f:
        arr = {k: np.asarray(f[k]) for k in f.files}
    e1 = arr['e1'].astype(np.float64)
    n = len(e1)
    assert n == seg['T'] and n > 0 and np.isfinite(e1).all()
    assert abs(float(e1.max())-seg['e1_max']) <= 1e-12, 'Saved array differs from V15 B0_RESULT'
    nq = NQ[task]
    start = seg['first_row'] if 'first_row' in seg else None
    rows = slice(start, start+n)
    obs = data['observations'][rows].astype(np.float64)
    nxt = data['next_observations'][rows].astype(np.float64)
    act = data['actions'][rows].astype(np.float64)
    clip_obs = (np.abs(obs[:, nq-1:]) >= 10-1e-6).any(axis=1)
    clip_next = (np.abs(nxt[:, nq-1:]) >= 10-1e-6).any(axis=1)
    out_of_box = (np.abs(act) > 1).any(axis=1)
    boundary = arr['boundary'].astype(bool)
    terminal = arr['terminal'].astype(bool)
    reference = arr['reference_terminal'].astype(bool)
    mismatch = np.flatnonzero(terminal != reference)
    firsts = {str(level): first_above(e1, level) for level in cfg['levels']}
    gate_step = firsts[str(cfg['B0_normalized_residual_limit'])]
    window = cfg['window_steps']
    growth = []
    for a in range(0, n, window):
        e = e1[a:a+window]
        growth.append(dict(steps='%d-%d' % (a+1, min(a+window, n)), n=int(len(e)), median=pct(e, 50),
                           p90=pct(e, 90), max=float(e.max()), clipped_next_velocity_steps=int(clip_next[a:a+window].sum()),
                           action_on_bound_steps=int(boundary[a:a+window].sum())))
    near = slice(max(0, (gate_step or 1)-1-cfg['context_steps']), (gate_step or 0)+cfg['context_steps']) if gate_step else slice(0, 0)
    return dict(trajectory_id=seg['trajectory_id'], first_row=start, L=seg['L'], T=n,
        reached_record_end=seg['reached_record_end'], passed_v15=seg['passed'],
        first_step_above=firsts, first_step_above_gate=gate_step,
        steps_within_gate_from_start=(gate_step-1) if gate_step else n,
        terminal_mismatch_steps=(mismatch+1).tolist()[:20],
        recorded_clipped_velocity_steps=int(clip_obs.sum()), recorded_clipped_next_velocity_steps=int(clip_next.sum()),
        first_recorded_clipped_next_velocity_step=first_above(clip_next.astype(float), .5),
        recorded_actions_outside_box_steps=int(out_of_box.sum()),
        tracking_action_on_bound_steps=int(boundary.sum()), first_tracking_action_on_bound_step=first_above(boundary.astype(float), .5),
        solver_iterations_max=int(arr['iterations'].max()), action_delta_max=float(arr['action_delta'].max()),
        near_first_exceed=dict(steps=[int(i)+1 for i in range(n)][near], e1=e1[near].tolist(),
            clipped_next_velocity=clip_next[near].tolist(), action_on_bound=boundary[near].tolist()),
        growth=growth, array_sha256=sha(npz))


def fmt(x, digits=4):
    return '无' if x is None else ('%.*g' % (digits, x) if isinstance(x, float) else str(x))


def table(header, rows):
    return '\n'.join(['|'+'|'.join(header)+'|', '|'+'|'.join(['---']*len(header))+'|']
                     + ['|'+'|'.join(fmt(c) for c in r)+'|' for r in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=None, help='V15 output directory name (default from config.json)')
    args = parser.parse_args()
    for name, digest in read(HERE/'FROZEN.json')['files'].items():
        assert sha(HERE/name) == digest, ('V15b package hash mismatch', name)
    cfg = read(HERE/'config.json')
    run = V15/'outputs'/(args.run or cfg['v15_run'])
    assert (run/'COMPLETED.json').is_file(), 'V15 run not found or not completed: '+str(run)
    import h5py
    spec = read(PROJECT/'cluster/paths.example.json')['tasks']
    out = HERE/'outputs'/('b0diag_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True)
    (HERE/'outputs/LATEST_RUN.txt').write_text(str(out)+'\n', encoding='utf-8')
    inputs, results = {}, {}
    for task in cfg['tasks']:
        tdir = run/task
        b0 = read(tdir/'B0_RESULT.json')
        plan = read(tdir/'PLAN.json')
        inputs[str(tdir/'B0_RESULT.json')] = sha(tdir/'B0_RESULT.json')
        inputs[str(tdir/'PLAN.json')] = sha(tdir/'PLAN.json')
        path = PROJECT/spec[task]['target_dataset']
        inputs[str(path)] = sha(path)
        with h5py.File(path, 'r') as f:
            data = {k: np.asarray(f[k]) for k in ('observations', 'next_observations', 'actions')}
        segments = {s['trajectory_id']: s for s in plan['target_segments']}
        rows = []
        for seg in b0['trajectories']:
            seg = dict(seg, first_row=segments[seg['trajectory_id']]['first_row'])
            npz = tdir/('B0_trajectory%05d.npz' % seg['trajectory_id'])
            rows.append(segment_report(task, seg, npz, data, cfg))
        within = [r['steps_within_gate_from_start'] for r in rows]
        results[task] = dict(segments=rows, B0_passed_v15=b0['passed'],
            min_steps_within_gate_over_segments=int(min(within)),
            min_steps_within_gate_over_failed_segments=None if all(r['passed_v15'] for r in rows)
                else int(min(r['steps_within_gate_from_start'] for r in rows if not r['passed_v15'])))
    write(out/'RESULT.json', dict(v15_run=str(run), config=cfg, tasks=results, input_sha256=inputs,
        simulator_steps=0, training_updates=0, GPU_jobs=0))
    text = ['# V15b B0 自检只读诊断', '',
        '不仿真、不训练、不占 GPU。读取 V15 已保存的 B0 逐步数组（`%s`）和目标数据。' % run,
        '“首次超限步”是从第 1 步起，归一化跟踪残差第一次超过该水平的执行步（1 起计）。', '']
    for task, res in results.items():
        segs = res['segments']
        text += ['## %s' % task, '',
            table(['段', 'L', 'T', 'V15结果'] + ['首次>%s' % l for l in cfg['levels']] + ['终止不一致首步', '记录中速度截断步数', '跟踪动作触边步数', '记录动作越界步数'],
                [[s['trajectory_id'], s['L'], s['T'], '通过' if s['passed_v15'] else '不通过']
                 + [s['first_step_above'][str(l)] for l in cfg['levels']]
                 + [s['terminal_mismatch_steps'][0] if s['terminal_mismatch_steps'] else None,
                    s['recorded_clipped_next_velocity_steps'], s['tracking_action_on_bound_steps'], s['recorded_actions_outside_box_steps']]
                 for s in segs]), '',
            '所有段从第 1 步起都保持在门槛 %s 以内的最短步数：%d；仅看不通过的段：%s。' % (
                fmt(cfg['B0_normalized_residual_limit']), res['min_steps_within_gate_over_segments'],
                fmt(res['min_steps_within_gate_over_failed_segments'])), '']
        for s in segs:
            if s['passed_v15']:
                continue
            text += ['### %s 第 %d 段：残差沿轨迹的增长（每 %d 步一档）' % (task, s['trajectory_id'], cfg['window_steps']), '',
                table(['步', 'n', '中位数', 'P90', '最大', '记录速度截断步数', '跟踪动作触边步数'],
                    [[g['steps'], g['n'], g['median'], g['p90'], g['max'], g['clipped_next_velocity_steps'], g['action_on_bound_steps']] for g in s['growth']]), '',
                '首次超过门槛前后 %d 步：' % cfg['context_steps'], '',
                table(['步', '残差', '记录下一观测速度截断', '跟踪动作触边'],
                    list(zip(s['near_first_exceed']['steps'], s['near_first_exceed']['e1'],
                             s['near_first_exceed']['clipped_next_velocity'], s['near_first_exceed']['action_on_bound']))), '']
    text += ['## 说明', '', '本诊断只描述，不作判定，也不提出下一步实验。逐段完整数据见 RESULT.json。', '']
    (out/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')
    print('V15b DONE:', out/'REPORT_ZH.md', flush=True)


if __name__ == '__main__':
    main()
