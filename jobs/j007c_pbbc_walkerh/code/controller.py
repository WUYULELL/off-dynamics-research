"""j007 controller (one task per sub-job): preparation, smoke runs, 25 screening runs, per-task summary and report.

Uses only the GPUs the relay put in CUDA_VISIBLE_DEVICES (one worker process sees exactly one of them),
at most max_per_gpu workers per GPU. Reads the transfer project, the ODRL archive and the installed
mujoco_py read-only; writes only under this job's code/ folder (private_runtime/, data/, runs_*/, outputs/).
Failure policy (as V21): a worker killed by an outside signal is rerun from step 0 with the same seed in a new
folder (at most 3 times); a code error / NaN stops new launches, running workers finish, exit code 1.
Rerun decisions never depend on returns.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
CFG_JOB = json.loads((HERE/'config.json').read_text(encoding='utf-8'))['job']
OUT = HERE/'outputs'
HOME = Path.home()
PROJECT = HOME/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
ALGO = PROJECT/'source_snapshot/algo'
ARCHIVE = HOME/'odrl_target_20261115.zip'
ARCHIVE_SHA = 'df9c2b43c0a1f479c2ebeac1a3bbf156a95fa8aaedc6e92d36443baea85ae6fc'
LOCAL_TEST = os.environ.get('J007_LOCAL_TEST')          # synthetic local test of the whole controller
ROBOTS = {'hopper': dict(env=('gym.envs.mujoco.hopper_v3', 'HopperEnv'), source_task='hopper-morph', sd=11, ad=3),
          'walker2d': dict(env=('gym.envs.mujoco.walker2d_v3', 'Walker2dEnv'), source_task='walker2d-morph', sd=17, ad=6)}
NAMES = {'T': 'T（只用目标数据的 IQL）', 'BC': 'BC（只用目标数据）', 'BCMIX': 'BC-混合（目标 + 源数据）',
         'PB25': 'PB25（候选：近邻加采样 BC-混合，源批中 16/64 来自近邻集）', 'PB50': 'PB50（候选：近邻加采样 BC-混合，源批中 32/64 来自近邻集）'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(str(tmp), str(path))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(*a):
    print('[%s %s]' % (CFG_JOB, utc()), *a, flush=True)


def prepare(cfg):
    """Frozen hashes, private mujoco_py, target data, source/learner hashes. Returns (vendor, algo, data, sources)."""
    spec = read(HERE/'FROZEN.json')
    bad = [n for n, d in spec['files'].items() if sha(HERE/n) != d]
    assert not bad, 'FROZEN hash mismatch: %s' % bad
    if LOCAL_TEST:
        t = read(LOCAL_TEST)
        return t['vendor'], t['algo'], Path(t['data']), t['sources'], t['source_sha']
    mujoco = HOME/'.mujoco210/mujoco210'
    assert (mujoco/'bin').is_dir(), 'missing %s' % mujoco
    from importlib import metadata
    original = Path(metadata.distribution('mujoco-py').locate_file('mujoco_py')).resolve()
    vendor = HERE/'private_runtime'
    if not (vendor/'mujoco_py').is_dir():
        vendor.mkdir(exist_ok=True)
        shutil.copytree(str(original), str(vendor/'mujoco_py'))
    sys.path.insert(0, str(vendor))
    import mujoco_py                                     # builds / loads once here, before workers start
    assert vendor.resolve() in Path(mujoco_py.__file__).resolve().parents, 'mujoco_py not imported from private copy'
    assert ARCHIVE.is_file() and sha(ARCHIVE) == ARCHIVE_SHA, 'ODRL archive missing or hash mismatch'
    data = HERE/'data'
    data.mkdir(exist_ok=True)
    needed = {t['target_file']: t['target_sha256'] for t in cfg['tasks']}
    with zipfile.ZipFile(str(ARCHIVE)) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if name in needed and '/raw/' in info.filename and not (data/name).exists():
                (data/name).write_bytes(z.read(info))
    for name, digest in needed.items():
        assert (data/name).is_file() and sha(data/name) == digest, 'target file %s' % name
    files = {str((PROJECT/e['path']).resolve()): e for e in read(PROJECT/'MANIFEST.json')['files']}
    paths_spec = read(PROJECT/'cluster/paths.example.json')['tasks']
    sources, source_sha = {}, {}
    for p in [ALGO/'offline_offline/iql_dyn.py', ALGO/'offline_offline/iql_unc.py']:
        e = files[str(p.resolve())]
        assert sha(p) == e['sha256'], str(p)
    for robot, r in ROBOTS.items():
        p = PROJECT/paths_spec[r['source_task']]['source_dataset']
        e = files[str(p.resolve())]
        d = sha(p)
        assert d == e['sha256'], str(p)
        sources[robot], source_sha[robot] = str(p), d
    return str(vendor), str(ALGO), data, sources, source_sha


def plan(cfg, vendor, algo, data, sources, source_sha, smoke):
    jobs = []
    tasks = [t for t in cfg['tasks'] if t['id'] == cfg['smoke']['task']] if smoke else cfg['tasks']
    seeds = [cfg['smoke']['seed']] if smoke else cfg['seeds']
    for group in cfg['groups']:                        # T first: the longest runs start early
        learner = 'IQL' if group == 'T' else 'BC'
        for task in tasks:
            robot = ROBOTS[task['robot']]
            for seed in seeds:
                name = '%s%s_%s_s%d' % ('smoke_' if smoke else '', group, task['id'], seed)
                job = dict(id=name, task=task['id'], group=group, learner=learner, seed=seed, seed_origin=seeds[0],
                           steps=cfg['smoke']['steps'] if smoke else cfg['steps'][learner],
                           eval_every=cfg['smoke']['steps'] if smoke else cfg['eval_every'][learner],
                           state_dim=robot['sd'], action_dim=robot['ad'], target_rows=cfg['target_rows'],
                           score_min=task['score_min'], score_max=task['score_max'],
                           xml=str(HERE/'assets'/task['xml']), xml_sha256=task['xml_sha256'],
                           target_dataset=str(data/task['target_file']), target_dataset_sha256=task['target_sha256'],
                           source_dataset=sources[task['robot']], source_dataset_sha256=source_sha[task['robot']],
                           env_module=robot['env'][0], env_class=robot['env'][1], vendor=vendor, algo=algo, smoke=smoke)
                if LOCAL_TEST:
                    job.update(read(LOCAL_TEST)['job_overrides'])
                jobs.append(job)
    return jobs


def visible_gpus():
    value = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    gpus = [g.strip() for g in value.split(',') if g.strip() and g.strip() != '-1']
    assert gpus, 'CUDA_VISIBLE_DEVICES is empty or -1: this job needs the GPU(s) the relay assigns'
    return gpus


def run_queue(cfg, jobs, base, gpus, phase):
    pending = [dict(j) for j in jobs]
    running, done, failed, events = {}, {}, {}, []
    last_beat = 0.
    while pending or running:
        for name, r in list(running.items()):
            code = r['process'].poll()
            if code is None:
                continue
            r['log'].close()
            del running[name]
            ok = code == 0 and (r['folder']/'COMPLETED.json').is_file()
            events.append(dict(time=utc(), event='exit', job=name, gpu=r['gpu'], attempt=r['attempt'], returncode=code, completed=ok))
            if ok:
                done[name] = r['folder']
            elif code < 0 and r['attempt'] < cfg['external_retry_max']:
                log('external interruption, rerun from step 0:', name, 'code', code)
                pending.insert(0, dict(r['job'], _attempt=r['attempt']+1))
            else:
                failed[name] = dict(folder=str(r['folder']), returncode=code, attempt=r['attempt'])
                tail = (r['folder']/'stdout.log').read_text(errors='replace').splitlines()[-40:]
                log('RUN FAILED', name, 'code', code, '\n'+'\n'.join(tail))
        if failed:
            pending = []
        while pending:
            load = {g: sum(1 for r in running.values() if r['gpu'] == g) for g in gpus}
            gpu = min(gpus, key=lambda g: load[g])
            if load[gpu] >= cfg['max_per_gpu']:
                break
            job = pending.pop(0)
            attempt = job.pop('_attempt', 0)
            folder = base/(job['id'] if attempt == 0 else '%s_retry%d' % (job['id'], attempt))
            folder.mkdir(parents=True)
            job = dict(job, gpu=gpu, output=str(folder), attempt=attempt)
            write(folder/'job.json', job)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            out = (folder/'stdout.log').open('w')
            worker = read(LOCAL_TEST)['worker'] if LOCAL_TEST else str(HERE/'worker.py')
            process = subprocess.Popen([sys.executable, '-u', '-B', worker, str(folder/'job.json')],
                                       stdout=out, stderr=subprocess.STDOUT, env=env, cwd=str(folder))
            running[job['id']] = dict(job=job, process=process, gpu=gpu, folder=folder, log=out, attempt=attempt)
            events.append(dict(time=utc(), event='start', job=job['id'], gpu=gpu, attempt=attempt))
            if not LOCAL_TEST:
                time.sleep(cfg['launch_gap_seconds'])
        if time.time()-last_beat > 60:
            last_beat = time.time()
            progress = {}
            for name, r in running.items():
                try:
                    p = read(r['folder']/'progress.json')
                    progress[name] = '%d/%d' % (p['step'], p['total'])
                except Exception:
                    progress[name] = 'starting'
            write(OUT/'progress.json', dict(time=utc(), phase=phase, pending=len(pending), running=len(running),
                                            done=len(done), failed=len(failed), progress=progress))
            log(phase, 'pending', len(pending), 'running', len(running), 'done', len(done), 'failed', len(failed))
        time.sleep(0.5 if LOCAL_TEST else 10)
    write(OUT/('events_%s.json' % phase), events)
    return done, failed


def stats(x):
    n = len(x)
    mean = sum(x)/n
    sd = math.sqrt(sum((v-mean)**2 for v in x)/(n-1)) if n > 1 else 0.
    return mean, sd, sd/math.sqrt(n)


def collect(folder, job):
    completed = read(folder/'COMPLETED.json')
    assert completed['steps'] == job['steps'] and completed['finite']
    points = list(range(job['eval_every'], job['steps']+1, job['eval_every']))
    evals = [read(folder/('eval_%07d.json' % p)) for p in points]
    origin = 500000+10*(job['seed']-job['seed_origin'])
    for p, e in zip(points, evals):
        assert e['step'] == p and len(e['scores']) == 10 and e['env_seeds'] == list(range(origin, origin+10))
        assert all(math.isfinite(s) for s in e['scores']) and abs(sum(e['scores'])/10-e['mean']) < 1e-9
    last = evals[-5:]
    lengths = [l for e in last for l in e['episode_lengths']]
    return dict(last5=sum(e['mean'] for e in last)/len(last), final=evals[-1]['mean'],
                best_point=max(e['mean'] for e in evals), length_mean=sum(lengths)/len(lengths),
                length_min=min(lengths), fraction_under_200=sum(l < 200 for l in lengths)/len(lengths),
                curve=[round(e['mean'], 3) for e in evals], eval_conditioning=completed.get('eval_conditioning'),
                wall_seconds=completed['wall_seconds'])


def paired(a, b):
    diff = [x-y for x, y in zip(a, b)]
    mean, sd, se = stats(diff)
    return dict(mean=mean, sd=sd, se=se, positive=sum(d > 0 for d in diff), diffs=diff)


def summarize(cfg, jobs, done):
    table = {}
    runs = []
    for job in jobs:
        r = collect(done[job['id']], job)
        table[(job['task'], job['group'], job['seed'])] = r
        runs.append(dict(task=job['task'], group=job['group'], seed=job['seed'], **{k: v for k, v in r.items() if k != 'curve'}))
    seeds = cfg['seeds']
    tasks = {}
    for task in [t['id'] for t in cfg['tasks']]:
        groups = {}
        for g in cfg['groups']:
            v = [table[(task, g, s)]['last5'] for s in seeds]
            mean, sd, se = stats(v)
            lengths = [table[(task, g, s)]['length_mean'] for s in seeds]
            groups[g] = dict(per_seed=dict(zip(map(str, seeds), v)), mean=mean, sd=sd, se=se, worst=min(v),
                             length_mean=sum(lengths)/len(lengths),
                             fraction_under_200=sum(table[(task, g, s)]['fraction_under_200'] for s in seeds)/len(seeds))
        strong = max(cfg['controls'], key=lambda g: groups[g]['mean'])
        cands = {}
        for c in cfg['candidates']:
            vs = paired(list(groups[c]['per_seed'].values()), list(groups[strong]['per_seed'].values()))
            cands[c] = dict(minus_strong=vs, screen_effective=vs['mean'] > 0 and vs['mean'] > vs['se'],
                            two_se_descriptive=vs['mean'] > 0 and vs['mean'] > 2*vs['se'],
                            minus_BCMIX=paired(list(groups[c]['per_seed'].values()), list(groups['BCMIX']['per_seed'].values())),
                            minus_BC=paired(list(groups[c]['per_seed'].values()), list(groups['BC']['per_seed'].values())),
                            minus_T=paired(list(groups[c]['per_seed'].values()), list(groups['T']['per_seed'].values())))
        tasks[task] = dict(groups=groups, strong_control=strong, candidates=cands,
                           BCMIX_minus_BC=paired(list(groups['BCMIX']['per_seed'].values()), list(groups['BC']['per_seed'].values())),
                           PB50_minus_PB25=paired(list(groups['PB50']['per_seed'].values()), list(groups['PB25']['per_seed'].values())))
    verdicts = {}                                      # the verdict needs all three sub-jobs; made in analysis/
    for c in cfg['candidates']:
        verdicts[c] = {t: dict(screen_effective=tasks[t]['candidates'][c]['screen_effective'],
                               nonpositive=tasks[t]['candidates'][c]['minus_strong']['mean'] <= 0) for t in tasks}
    return runs, tasks, verdicts


def fmt(x, d=2):
    return '' if x is None else ('%.*f' % (d, x) if isinstance(x, float) else str(x))


def report(cfg, tasks, verdicts, elapsed, prox):
    lines = ['# %s：近邻加采样 BC-混合（PB25、PB50）筛查（单任务子任务）' % cfg['job'], '',
             '种子 %s（新种子）；T、BC、BC-混合在同一台机器、同一批种子下重跑。主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。'
             % cfg['seeds'], '',
             '本任务是 j007 的一个子任务（j007a/b/c 各一个任务）。支持/否定的判定需要三个子任务的结果，在 analysis/ 中合并后按 research/PLAN.md 的登记规则做出；这里只给出本任务上每个候选是否“筛查有效”。', '',
             '## 预设规则（提交前写进 research/PLAN.md）', '']
    lines += ['- %s：%s' % (k, v) for k, v in cfg['rule'].items()]
    lines += ['', '## 本任务上的筛查结果', '']
    for c, v in verdicts.items():
        for t, r in v.items():
            lines.append('- %s：筛查有效 %s；均值差 ≤ 0：%s' % (c, '是' if r['screen_effective'] else '否', '是' if r['nonpositive'] else '否'))
    lines += ['', '## 近邻集的统计（描述性，各种子相同）', '']
    for g, info in prox.items():
        lines.append('- %s：阈值 tau=%.3f；近邻集 %d / %d 行（%.2f%%），每批 %d/64 个源行来自近邻集；目标留一轨迹最近距离分位数 %s；源数据到目标最近距离分位数 %s' % (
            g, info['tau'], info['near_rows'], info['source_rows'], 100*info['near_fraction'], info['near_rows_per_batch'],
            json.dumps(info['target_loto_quantiles']), json.dumps(info['source_distance_quantiles'])))
    for task, t in tasks.items():
        lines += ['', '## %s（强对照：%s，%.2f）' % (task, t['strong_control'], t['groups'][t['strong_control']]['mean']), '',
                  '|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|', '|---|---|---|---|---|---|---|']
        for g, r in t['groups'].items():
            lines.append('|%s|%.2f|%.2f|%.2f|%.1f|%.2f|%s|' % (NAMES[g], r['mean'], r['sd'], r['worst'], r['length_mean'],
                         r['fraction_under_200'], ' / '.join('%.1f' % x for x in r['per_seed'].values())))
        lines += ['', '|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|',
                  '|---|---|---|---|---|---|---|---|---|']
        for c, r in t['candidates'].items():
            m = r['minus_strong']
            lines.append('|%s|%.2f|%.2f|%d|%s|%s|%.2f（SE %.2f）|%.2f（SE %.2f）|%.2f（SE %.2f）|' % (
                c, m['mean'], m['se'], m['positive'], '是' if r['screen_effective'] else '否', '是' if r['two_se_descriptive'] else '否',
                r['minus_BCMIX']['mean'], r['minus_BCMIX']['se'], r['minus_BC']['mean'], r['minus_BC']['se'],
                r['minus_T']['mean'], r['minus_T']['se']))
        b = t['BCMIX_minus_BC']
        q = t['PB50_minus_PB25']
        lines.append('')
        lines.append('描述性：BC-混合 − BC %.2f（SE %.2f，%d/5 正）；PB50 − PB25 %.2f（SE %.2f，%d/5 正）。' % (
            b['mean'], b['se'], b['positive'], q['mean'], q['se'], q['positive']))
    lines += ['', '## 运行情况', '', '总耗时 %.1f 小时。冒烟运行的回报不读取、不报告。所有种子都保留。' % (elapsed/3600)]
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    started = time.time()
    OUT.mkdir(exist_ok=True)
    cfg = read(HERE/'config.json')
    gpus = ['cpu'] if LOCAL_TEST else visible_gpus()
    log('GPUs', gpus)
    vendor, algo, data, sources, source_sha = prepare(cfg)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    base = HERE/('runs_'+stamp)
    write(OUT/'STATUS.json', dict(time=utc(), phase='smoke', runs_dir=str(base), gpus=gpus))
    smoke = plan(cfg, vendor, algo, data, sources, source_sha, True)
    done, failed = run_queue(cfg, smoke, base/'smoke', gpus, 'smoke')
    if failed:
        write(OUT/'STATUS.json', dict(time=utc(), phase='smoke_failed', failed=failed))
        log('smoke failed', failed)
        return 1
    for job in smoke:
        c = read(done[job['id']]/'COMPLETED.json')
        assert c['finite'] and c['evaluations'] == 1, job['id']
    log('smoke passed', len(done))
    jobs = plan(cfg, vendor, algo, data, sources, source_sha, False)
    write(OUT/'PLAN.json', [dict(id=j['id'], task=j['task'], group=j['group'], seed=j['seed'], steps=j['steps']) for j in jobs])
    write(OUT/'STATUS.json', dict(time=utc(), phase='main', runs=len(jobs), runs_dir=str(base)))
    done, failed = run_queue(cfg, jobs, base/'main', gpus, 'main')
    if failed:
        write(OUT/'STATUS.json', dict(time=utc(), phase='main_failed', failed=failed, completed=len(done)))
        log('main phase has failed runs; no summary', failed)
        return 1
    runs, tasks, verdicts = summarize(cfg, jobs, done)
    with (OUT/'RUNS.csv').open('w', newline='', encoding='utf-8') as f:
        keys = ['task', 'group', 'seed', 'last5', 'final', 'best_point', 'length_mean', 'length_min',
                'fraction_under_200', 'eval_conditioning', 'wall_seconds']
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in runs:
            w.writerow(r)
    curves = {j['id']: collect(done[j['id']], j)['curve'] for j in jobs}
    write(OUT/'CURVES.json', curves)
    write(OUT/'summary.json', dict(job=cfg['job'], seeds=cfg['seeds'], rule=cfg['rule'], tasks=tasks, per_task_screen=verdicts,
                                   elapsed_hours=(time.time()-started)/3600))
    prox = {}
    for j in jobs:
        if j['group'] in cfg['candidates'] and j['group'] not in prox:
            prox[j['group']] = read(done[j['id']]/'COMPLETED.json')['source']['proximity']
    report(cfg, tasks, verdicts, time.time()-started, prox)
    write(OUT/'STATUS.json', dict(time=utc(), phase='completed', runs=len(jobs)))
    print((OUT/'REPORT_ZH.md').read_text(encoding='utf-8'), flush=True)
    return 0


if __name__ == '__main__':
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        try:
            write(OUT/'STATUS.json', dict(time=utc(), phase='controller_error', traceback=traceback.format_exc()[-3000:]))
        except Exception:
            pass
        code = 1
    sys.exit(code)
