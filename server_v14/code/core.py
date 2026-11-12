"""CPU-only diagnostics: shared numerical definitions, no optimizer or training."""
import csv
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MUJOCO_PY_FORCE_CPU'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = 'hopper-morph'
VARIANTS = ('BC-all','BC-fix','BC-mix','BC-mix-sel')
SEEDS = list(range(601,606))
A_EDGES = [0,50,100,150,200,400]
A_LABELS = ['0-49','50-99','100-149','150-199','200-399','>=400']


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, replace=False):
    path = Path(path)
    if not replace and path.exists():
        raise FileExistsError(str(path))
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    os.replace(str(tmp),str(path))


def csv_write(path, rows):
    assert rows
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name,str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stats(values):
    a = np.asarray(values,dtype=np.float64).reshape(-1)
    if not len(a):
        return dict(n=0,mean=None,median=None,min=None,p10=None,p90=None,p99=None,max=None)
    if not np.isfinite(a).all():
        raise FloatingPointError('Nonfinite diagnostic values')
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),
                p10=float(np.percentile(a,10)),p90=float(np.percentile(a,90)),
                p99=float(np.percentile(a,99)),max=float(a.max()))


def bins(steps, edges=A_EDGES):
    return np.searchsorted(np.asarray(edges),np.asarray(steps),side='right')-1


def segments(data, tolerance=1e-6, source=False):
    """Raw chronological rows, terminal/timeout boundaries, explicit-next gaps.

    A source complete episode must start just after a recorded terminal/timeout,
    end at one, and contain no gap. Unknown data head/tail are not called complete.
    Target B0 uses every available segment, including the recorded partial tail.
    """
    obs, nxt = data['observations'],data['next_observations']
    n = len(obs)
    terminal = data['terminals'].reshape(-1).astype(bool)
    timeout = data.get('timeouts',np.zeros(n,dtype=bool)).reshape(-1).astype(bool)
    event = terminal|timeout
    gap = np.r_[np.max(np.abs(nxt[:-1]-obs[1:]),axis=1)>tolerance,False]
    ends = np.flatnonzero(event|gap|np.r_[np.zeros(n-1,dtype=bool),True])
    records, step = [],np.empty(n,dtype=np.int64)
    start = 0
    for number,end in enumerate(ends):
        end = int(end)
        known_start = start>0 and bool(event[start-1])
        complete = bool(known_start and event[end]) if source else bool(event[end] or gap[end])
        records.append(dict(trajectory_id=number,first_row=start,last_row=end,length=end-start+1,
                            complete=complete,known_reset_start=known_start,
                            end_terminal=bool(terminal[end]),end_timeout=bool(timeout[end]),
                            end_gap=bool(gap[end])))
        step[start:end+1] = np.arange(end-start+1)
        start = end+1
    assert start==n and sum(r['length'] for r in records)==n
    return records,step


def choose_episodes(records, count, seed):
    complete = [r for r in records if r['complete']]
    assert complete, 'No complete source trajectories'
    order = np.random.RandomState(seed).choice(len(complete),min(count,len(complete)),replace=False)
    return [complete[int(i)] for i in order],len(complete)


def b0_gate(raw_linf, actual_terminal, reference_terminal, expected_length, tolerance):
    error = np.asarray(raw_linf)
    actual = np.asarray(actual_terminal,dtype=bool)
    reference = np.asarray(reference_terminal,dtype=bool)[:len(actual)]
    assert len(error)==len(actual)>0 and np.isfinite(error).all()
    over = np.flatnonzero(error>tolerance)
    mismatch = np.flatnonzero(actual!=reference)
    reached = len(error)==expected_length
    return dict(passed=bool(reached and not len(over) and not len(mismatch)),reached_record_end=reached,
                first_exceed_step=None if not len(over) else int(over[0])+1,
                terminal_mismatch_steps=(mismatch+1).tolist(),raw_linf=stats(error))


def interpret_a(mix, all_):
    assert np.isfinite([mix,all_]).all() and mix>=0 and all_>=0
    if mix<=2*all_:
        return '目标数据已被拟合；结合 A3 描述实际轨迹是否离开目标数据区域。'
    if mix>5*all_:
        return '目标项基本未被拟合；逐项列明采样、损失与优化的代码和记录核查结果。'
    return '不确定：处于预设 2 倍与 5 倍边界之间。'


def interpret_b(lengths, ratios):
    lengths,ratios = np.asarray(lengths),np.asarray(ratios)
    fraction = float((ratios>=.8).mean())
    if np.median(lengths)<200 and fraction<.1:
        return '不可跟踪：仅限当前一步贪心跟踪方式，不能排除前瞻控制器更好。'
    if fraction>=.5:
        return '可跟踪：按预设描述，状态轨迹可行；阶段 1 失败应归于学习或数据使用方式。'
    return '不确定。'


def histogram_svg(path, counts_by_mode):
    """20-step equal-width termination counts, zero-based y axis, no smoothing."""
    from xml.sax.saxutils import escape
    width,height = 1100,210*len(counts_by_mode)+70
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="70" y="28" font-family="sans-serif" font-size="18">Physical termination step (equal 20-step bins)</text>']
    for panel,(mode,counts) in enumerate(counts_by_mode.items()):
        y = 60+panel*210
        maximum = max(1,max(counts))
        parts.append(f'<text x="70" y="{y+13}" font-family="sans-serif" font-size="14">{escape(mode)}; terminal episodes = {sum(counts)}</text>')
        parts.append(f'<path d="M70 {y+25} V{y+155} H1070" fill="none" stroke="black"/>')
        for j,c in enumerate(counts):
            h = c/maximum*120
            parts.append(f'<rect x="{70+20*j}" y="{y+155-h}" width="19" height="{h}" fill="#0072B2"/>')
        for j in (0,10,20,30,40,50):
            parts.append(f'<text x="{70+20*j}" y="{y+177}" text-anchor="middle" font-family="sans-serif" font-size="12">{20*j}</text>')
        parts.append(f'<text x="60" y="{y+159}" text-anchor="end" font-family="sans-serif" font-size="12">0</text>')
        parts.append(f'<text x="60" y="{y+40}" text-anchor="end" font-family="sans-serif" font-size="12">{maximum}</text>')
    parts.append('</svg>')
    Path(path).write_text('\n'.join(parts),encoding='utf-8')
