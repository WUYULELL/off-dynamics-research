"""V18 controller: pilot A, does mixing D4RL medium source data help on shifts that leave it valid?

Plans the runs, starts them on GPUs that nobody else is using, retries only
externally killed runs, then audits the outputs on CPU and writes the report.
Reads the transfer project and $HOME/v13_action_fix/stage0 read-only; writes
only under $HOME/v18_pilotA (outputs/ and data/).
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
PROJECT = Path.home()/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
V13 = Path.home()/'v13_action_fix'
ALGO = PROJECT/'source_snapshot/algo'
ROBOTS = {'hopper': dict(env=('gym.envs.mujoco.hopper_v3', 'HopperEnv'), source_task='hopper-morph', state_dim=11, action_dim=3),
          'walker2d': dict(env=('gym.envs.mujoco.walker2d_v3', 'Walker2dEnv'), source_task='walker2d-morph', state_dim=17, action_dim=6)}
GROUPS = ('T', 'BC', 'MIXVQ', 'MIXALL', 'BCMIX')
IQL_GROUPS = ('T', 'MIXVQ', 'MIXALL')
MIXED = ('MIXVQ', 'MIXALL', 'BCMIX')
DATA = HERE/'data'
TEST_WORKER = os.environ.get('V18_TEST_WORKER')      # synthetic local test of the controller only


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(str(path))
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(str(tmp), str(path))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def prepare_data(cfg):
    """Unpack the six ODRL target files once into this package's own data folder and verify the three used."""
    import zipfile
    archive = Path.home()/cfg['target_archive']
    if not DATA.is_dir():
        assert archive.is_file(), 'Upload %s to the home directory first' % cfg['target_archive']
        assert sha(archive) == cfg['target_archive_sha256'], 'Target archive hash mismatch'
        tmp = HERE/('data.tmp.'+uuid.uuid4().hex)
        with zipfile.ZipFile(str(archive)) as z:
            for info in z.infolist():
                name = Path(info.filename).name
                if info.filename.endswith('.hdf5') and '/raw/' in info.filename:
                    tmp.mkdir(exist_ok=True)
                    (tmp/name).write_bytes(z.read(info))
        os.replace(str(tmp), str(DATA))
    for task in cfg['tasks']:
        assert sha(DATA/task['target_file']) == task['target_sha256'], task['target_file']
        assert sha(HERE/'assets'/task['xml']) == task['xml_sha256'], task['xml']


def protected(cfg):
    manifest = {str((PROJECT/e['path']).resolve()): e for e in read(PROJECT/'MANIFEST.json')['files']}
    spec = read(PROJECT/'cluster/paths.example.json')['tasks']
    paths = [ALGO/'offline_offline/iql_dyn.py', ALGO/'offline_offline/iql_unc.py']
    paths += [PROJECT/spec[ROBOTS[r]['source_task']]['source_dataset'] for r in sorted({t['robot'] for t in cfg['tasks']})]
    hashes = {}
    for path in paths:
        entry = manifest[str(path.resolve())]
        digest = sha(path)
        assert entry['sha256'] == digest and path.stat().st_size == entry['bytes'], str(path)
        hashes[str(path)] = digest
    for task in cfg['tasks']:
        hashes[str(DATA/task['target_file'])] = sha(DATA/task['target_file'])
    return hashes


def smi(*args):
    return subprocess.run(['nvidia-smi']+list(args)+['--format=csv,noheader,nounits'], check=True,
                          stdout=subprocess.PIPE, universal_newlines=True, timeout=60).stdout.strip().splitlines()


def gpus():
    """Per card: utilization, memory, and the number of compute processes the driver lists on it.
    Driver PIDs are not comparable with PIDs inside this container, so only the count is used."""
    if TEST_WORKER:
        return {i: dict(uuid='test%d' % i, utilization=0, used_MiB=0, total_MiB=49140, processes=0) for i in range(4)}
    rows = {}
    for line in smi('--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total'):
        index, uuid_, util, used, total = [x.strip() for x in line.split(',')]
        rows[int(index)] = dict(uuid=uuid_, utilization=int(float(util)), used_MiB=int(float(used)),
                                total_MiB=int(float(total)), processes=0)
    by_uuid = {r['uuid']: r for r in rows.values()}
    for line in smi('--query-compute-apps=gpu_uuid,pid'):
        if line.strip():
            by_uuid[line.split(',')[0].strip()]['processes'] += 1
    return rows


def plan(cfg, out, vendor, smoke, source_hashes):
    spec = read(PROJECT/'cluster/paths.example.json')['tasks']
    jobs = []
    seeds = [cfg['smoke_seed']] if smoke else cfg['seeds']
    tasks = [t for t in cfg['tasks'] if t['id'] in cfg['smoke_tasks']] if smoke else cfg['tasks']
    for group in ('T', 'MIXVQ', 'MIXALL', 'BC', 'BCMIX'):          # the long IQL runs first
        learner = 'IQL' if group in IQL_GROUPS else 'BC'
        for task in tasks:
            robot = ROBOTS[task['robot']]
            source = PROJECT/spec[robot['source_task']]['source_dataset']
            for seed in seeds:
                name = '%s%s_%s_s%d' % ('smoke_' if smoke else '', group, task['id'], seed)
                jobs.append(dict(id=name, task=task['id'], group=group, learner=learner, seed=seed,
                    seed_origin=cfg['smoke_seed'] if smoke else cfg['seeds'][0],
                    steps=cfg['smoke_steps'] if smoke else cfg['steps'][learner],
                    eval_every=cfg['smoke_steps'] if smoke else cfg['eval_every'][learner],
                    state_dim=robot['state_dim'], action_dim=robot['action_dim'],
                    score_min=task['score_min'], score_max=task['score_max'],
                    xml=str(HERE/'assets'/task['xml']), xml_sha256=task['xml_sha256'],
                    target_dataset=str(DATA/task['target_file']), target_dataset_sha256=task['target_sha256'],
                    source_dataset=str(source), source_dataset_sha256=source_hashes[str(source)],
                    env_module=robot['env'][0], env_class=robot['env'][1], vendor=str(vendor), algo=str(ALGO),
                    smoke=smoke, base=str(out)))
    return jobs


class Queue:
    def __init__(self, cfg, out, heartbeat):
        self.cfg, self.out, self.heartbeat = cfg, out, heartbeat
        self.running, self.events, self.blocked = {}, [], {}

    def card(self):
        """A card receives a new run only if every compute process on it is one of ours and, when we have
        nothing on it, it is idle. Anything else means another user is there: leave the card alone."""
        state = gpus()
        best = None
        for index, row in sorted(state.items()):
            if index not in self.cfg['gpu_indices']:
                continue
            mine = sum(1 for r in self.running.values() if r['gpu'] == index)
            if mine >= self.cfg['max_per_gpu']:
                continue
            # More compute processes than our own runs, or activity on a card we are not using.
            foreign = row['processes'] > mine or (mine == 0 and (row['utilization'] > self.cfg['idle_utilization_max']
                                                               or row['used_MiB'] > self.cfg['idle_memory_max_MiB']))
            if foreign:
                b = self.blocked.setdefault(index, dict(checks=0, samples=[]))
                b['checks'] += 1
                if len(b['samples']) < 20 and (not b['samples'] or time.time()-b['samples'][-1]['t'] > 60):
                    b['samples'].append(dict(t=time.time(), time=utc(), own_runs=mine, driver_processes=row['processes'],
                                             utilization=row['utilization'], used_MiB=row['used_MiB']))
                continue
            if best is None or mine < best[1]:
                best = (index, mine)
        return None if best is None else best[0]

    def start(self, job, gpu, attempt):
        folder = self.out/'runs'/(job['id'] if attempt == 0 else '%s_retry%d' % (job['id'], attempt))
        folder.mkdir(parents=True)
        job = dict(job, gpu=gpu, output=str(folder), attempt=attempt)
        write(folder/'job.json', job)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        command = [sys.executable, '-u', '-B', TEST_WORKER or str(HERE/'v18_worker.py'), str(folder/'job.json')]
        log = (folder/'stdout.log').open('w')
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(folder))
        self.running[job['id']] = dict(job=job, process=process, gpu=gpu, folder=folder, log=log, attempt=attempt, started=time.time())
        self.events.append(dict(time=utc(), event='start', job=job['id'], gpu=gpu, attempt=attempt, pid=process.pid))

    def run(self, jobs, phase):
        pending = list(jobs)
        done, failed = {}, {}
        last_beat = 0.
        while pending or self.running:
            for name, r in list(self.running.items()):
                code = r['process'].poll()
                if code is None:
                    continue
                r['log'].close()
                del self.running[name]
                ok = code == 0 and (r['folder']/'COMPLETED.json').is_file()
                self.events.append(dict(time=utc(), event='exit', job=name, gpu=r['gpu'], attempt=r['attempt'], returncode=code, completed=ok))
                if ok:
                    done[name] = r['folder']
                elif code < 0 and r['attempt'] < self.cfg['external_retry_max']:
                    # Killed by a signal from outside: same seed and config, from step 0, in a new folder.
                    pending.insert(0, dict(r['job'], _attempt=r['attempt']+1))
                else:
                    failed[name] = dict(folder=str(r['folder']), returncode=code, attempt=r['attempt'])
            if failed:
                pending = []            # a code error stops new launches; running jobs finish
            while pending:
                gpu = self.card()
                if gpu is None:
                    break
                job = pending.pop(0)
                attempt = job.pop('_attempt', 0)
                self.start(job, gpu, attempt)
                time.sleep(self.cfg['launch_gap_seconds'])
            if time.time()-last_beat > 30:
                last_beat = time.time()
                progress = {}
                for name, r in self.running.items():
                    try:
                        p = read(r['folder']/'progress.json')
                        progress[name] = '%d/%d gpu%d' % (p['step'], p['total'], r['gpu'])
                    except Exception:
                        progress[name] = 'starting gpu%d' % r['gpu']
                self.heartbeat(phase, pending=len(pending), running=len(self.running), done=len(done),
                               failed=len(failed), progress=progress,
                               blocked_checks={str(i): b['checks'] for i, b in self.blocked.items()})
            time.sleep(2 if TEST_WORKER else 5)
        return done, failed


def stats(x):
    n = len(x)
    mean = sum(x)/n
    sd = math.sqrt(sum((v-mean)**2 for v in x)/(n-1)) if n > 1 else 0.
    return mean, sd


def collect(folder, job, cfg):
    completed = read(folder/'COMPLETED.json')
    assert completed['steps'] == job['steps'] and completed['finite']
    points = list(range(job['eval_every'], job['steps']+1, job['eval_every']))
    events = [read(folder/('eval_%07d.json' % p)) for p in points]
    origin = 500000+10*(job['seed']-job['seed_origin'])
    for p, e in zip(points, events):
        assert e['step'] == p and len(e['scores']) == 10 and e['env_seeds'] == list(range(origin, origin+10))
        assert all(math.isfinite(s) for s in e['scores']) and abs(sum(e['scores'])/10-e['mean']) < 1e-9
    last = events[-5:]
    lengths = [l for e in last for l in e['episode_lengths']]
    return dict(last5=sum(e['mean'] for e in last)/len(last), final=events[-1]['mean'],
                best_point=max(e['mean'] for e in events), length_mean=sum(lengths)/len(lengths),
                fraction_under_200=sum(l < 200 for l in lengths)/len(lengths),
                wall_seconds=completed['wall_seconds'], source=completed.get('source'))


def paired(a, b):
    diff = [x-y for x, y in zip(a, b)]
    mean, sd = stats(diff)
    return mean, sd/math.sqrt(len(diff)), sum(d > 0 for d in diff)


def summarize(cfg, jobs, done):
    table, rows, source_info = {}, [], {}
    for job in jobs:
        r = collect(done[job['id']], job, cfg)
        if r['source']:
            source_info[job['task']] = r['source']
        table[(job['task'], job['group'], job['seed'])] = r
        rows.append(dict(task=job['task'], group=job['group'], seed=job['seed'], **{k: v for k, v in r.items() if k != 'source'}))
    groups = []
    for task in [t['id'] for t in cfg['tasks']]:
        values = {g: [table[(task, g, s)]['last5'] for s in cfg['seeds']] for g in GROUPS}
        means = {g: stats(values[g])[0] for g in GROUPS}
        strong = 'T' if means['T'] >= means['BC'] else 'BC'
        for g in GROUPS:
            mean, sd = stats(values[g])
            row = dict(task=task, group=g, seeds=len(values[g]), last5_mean=mean, last5_sd=sd, worst_seed=min(values[g]),
                       final_mean=stats([table[(task, g, s)]['final'] for s in cfg['seeds']])[0],
                       length_mean=stats([table[(task, g, s)]['length_mean'] for s in cfg['seeds']])[0],
                       fraction_under_200=stats([table[(task, g, s)]['fraction_under_200'] for s in cfg['seeds']])[0],
                       strong_baseline=strong, strong_baseline_mean=means[strong])
            if g in MIXED:
                d, se, pos = paired(values[g], values[strong])
                row.update(minus_strong=d, minus_strong_SE=se, positive_seeds=pos, effective=bool(d > 0 and d > 2*se))
                for other in ('T', 'BC'):
                    d2, se2, pos2 = paired(values[g], values[other])
                    row.update({'minus_'+other: d2, 'minus_%s_SE' % other: se2})
            groups.append(row)
    return rows, groups, source_info


def fmt(x, digits=2):
    return '' if x is None else ('%.*f' % (digits, x) if isinstance(x, float) else str(x))


def table(header, rows):
    return '\n'.join(['|'+'|'.join(header)+'|', '|'+'|'.join(['---']*len(header))+'|']
                     + ['|'+'|'.join(fmt(c) for c in r)+'|' for r in rows])


def report(out, cfg, groups, source_info, queue, elapsed, smoke_seconds):
    names = dict(T='T（只用目标数据的 IQL）', BC='BC（只用目标数据）', MIXVQ='混合-价值（源数据只进 V、Q）',
                 MIXALL='混合-全部（源数据进 V、Q 和策略）', BCMIX='BC-混合（目标 + 源数据）')
    text = ['# V18 试点 A：在“偏移不影响源数据”的任务上，混合源数据能否超过只用目标数据', '',
        '任务来自 ODRL 的关节限位偏移。V16 筛查显示，这些偏移下 D4RL medium 源轨迹在目标仿真器中基本可以原样复现。',
        '源数据：D4RL medium（去掉超时行，奖励与下一状态按原记录，不做任何修正）；目标数据：ODRL 提供的 5000 条 medium 数据。分数按 ODRL 的参考回报归一化，不能与此前 OTDF 形态偏移任务的分数直接比较。',
        '本次输出：`%s`。' % out, '',
        '## 预设规则（结果产生前写定）', '',
        '- 主指标 Last5：最后五个评估点的平均（IQL 每 1 万步、BC 每 5000 步评估 10 回合）。种子 %s，各组同种子、共用初始化和目标数据采样序列。' % cfg['seeds'],
        '- 强基线：该任务上 T 与 BC 中 Last5 均值较高的一组。',
        '- 有效：混合组相对强基线的同种子配对差，均值 > 0 且 > 2 倍配对标准误。',
        '- 本次是 5 个种子的筛查；“有效”的组需用新种子确认后才能下结论。规则看到结果后不修改。', '']
    for task in cfg['tasks']:
        rows = [g for g in groups if g['task'] == task['id']]
        text += ['## %s（强基线：%s，%.2f）' % (task['id'], rows[0]['strong_baseline'], rows[0]['strong_baseline_mean']), '',
            'V16 中该任务：照搬源动作能跟到 80%% 长度的比例 %s，逐步跟踪 %s。' % (task['v16_B1'], task['v16_B2']), '',
            table(['组', 'Last5均值', 'SD', '最差种子', '相对强基线', '配对SE', '正号种子', '有效', '相对T', '相对BC', '回合长度', '<200步比例'],
                [[names[g['group']], g['last5_mean'], g['last5_sd'], g['worst_seed'], g.get('minus_strong'), g.get('minus_strong_SE'),
                  g.get('positive_seeds'), {True: '是', False: '否', None: ''}[g.get('effective')], g.get('minus_T'), g.get('minus_BC'),
                  g['length_mean'], g['fraction_under_200']] for g in rows]), '']
    retries = [e for e in queue.events if e['event'] == 'start' and e['attempt'] > 0]
    blocked = {str(i): b['checks'] for i, b in sorted(queue.blocked.items())}
    text += ['## 运行情况', '',
        '正式训练 %d 项，总耗时 %.1f 小时（另有冒烟测试 %.0f 秒，其结果不计入）。外部中断后自动重跑 %d 次。'
        % (len([e for e in queue.events if e['event'] == 'exit' and e['completed'] and not e['job'].startswith('smoke_')]), elapsed/3600, smoke_seconds, len(retries)),
        '各卡被判为“有他人进程或占用、暂不放行”的检查次数：%s（明细见 GPU_BLOCKED.json）。' % (blocked or '无'),
        '源数据行数（去掉超时行之后）：%s。' % json.dumps({k: v['rows'] for k, v in source_info.items()}, ensure_ascii=False),
        'hopper 的“难”与“易”两个目标数据文件在 ODRL 网盘上逐字节相同，无法确定归属，本次未使用。',
        '本次不提出也不启动下一项实验。', '']
    (out/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    for name, digest in read(HERE/'FROZEN.json')['files'].items():
        assert sha(HERE/name) == digest, ('V18 package hash mismatch', name)
    cfg = read(HERE/'config.json')
    if not TEST_WORKER:
        assert sys.platform == 'linux' and sys.version_info[:3] == (3, 8, 13)
        assert Path(sys.prefix).resolve() == (Path.home()/'conda-envs/offdynamics38').resolve()
    (HERE/'outputs').mkdir(exist_ok=True)
    lock = (HERE/'outputs/v18.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = HERE/'outputs'/('pilotA_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    (out/'runs').mkdir()
    (HERE/'outputs/LATEST_RUN.txt').write_text(str(out)+'\n', encoding='utf-8')
    print('V18 output:', out, flush=True)
    started = time.monotonic()

    def heartbeat(phase, **kw):
        write(out/'STATUS.json', dict(time=utc(), phase=phase, **kw), replace=True)
        print(phase, {k: v for k, v in kw.items() if k != 'progress'}, flush=True)

    queue = Queue(cfg, out, heartbeat)
    try:
        heartbeat('VERIFYING_INPUTS')
        prepare_data(cfg)
        before = protected(cfg)
        write(out/'PROTECTED_BEFORE.json', before)
        write(out/'config.json', cfg)
        write(out/'GPU_AT_START.json', gpus())
        if TEST_WORKER:
            vendor = out
        else:
            sys.path.insert(0, str(V13/'stage0'))
            import importlib.util
            spec = importlib.util.spec_from_file_location('v18_binding_loader', str(V13/'stage0/legacy_replay.py'))
            legacy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(legacy)
            binding, binding_before, private = legacy.prepare_private_binding(out)
            vendor = private.parent
            sys.path.insert(0, str(vendor))
            import mujoco_py                    # build the private copy once before any worker starts
            assert private.resolve() in Path(mujoco_py.cymj.__file__).resolve().parents
        smoke_jobs = plan(cfg, out, vendor, True, before)
        heartbeat('SMOKE_TEST', jobs=len(smoke_jobs))
        t0 = time.monotonic()
        done, failed = queue.run(smoke_jobs, 'SMOKE_TEST')
        smoke_seconds = time.monotonic()-t0
        if failed or len(done) != len(smoke_jobs):
            raise RuntimeError('Smoke test failed: %s' % json.dumps(failed, ensure_ascii=False))
        for job in smoke_jobs:
            completed = read(done[job['id']]/'COMPLETED.json')
            assert completed['steps'] == cfg['smoke_steps'] and completed['finite'] and completed['evaluations'] == 1
            assert (completed['source'] is not None) == (job['group'] in MIXED)
        write(out/'SMOKE_PASSED.json', dict(passed=True, jobs=len(smoke_jobs), seconds=smoke_seconds,
            note='Smoke runs use a separate seed and %d steps; their returns are not read or reported.' % cfg['smoke_steps']))
        jobs = plan(cfg, out, vendor, False, before)
        write(out/'PLAN.json', dict(jobs=[j['id'] for j in jobs], count=len(jobs)))
        heartbeat('TRAINING', jobs=len(jobs))
        done, failed = queue.run(jobs, 'TRAINING')
        write(out/'QUEUE_EVENTS.json', queue.events)
        write(out/'GPU_BLOCKED.json', {str(k): v for k, v in queue.blocked.items()})
        if failed or len(done) != len(jobs):
            raise RuntimeError('Training runs failed: %s' % json.dumps(failed, ensure_ascii=False))
        heartbeat('AUDITING')
        rows, groups, source_info = summarize(cfg, jobs, done)
        after = protected(cfg)
        assert before == after, 'Protected original inputs changed'
        if not TEST_WORKER:
            assert legacy.hash_tree(binding) == binding_before, 'Installed mujoco-py binding changed'
        elapsed = time.monotonic()-started
        import csv
        with (out/'RUNS.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        write(out/'RESULT.json', dict(groups=groups, runs=rows))
        report(out, cfg, groups, source_info, queue, elapsed, smoke_seconds)
        write(out/'COMPLETED.json', dict(completed=True, status='AWAITING_USER_REVIEW', runs=len(rows),
            effective=[(g['task'], g['group']) for g in groups if g.get('effective')],
            elapsed_seconds=elapsed, ended_utc=utc(),
            evidence_sha256={n: sha(out/n) for n in ('REPORT_ZH.md', 'RESULT.json', 'config.json')}))
        heartbeat('STOPPED_AWAITING_USER_REVIEW')
        print('V18 DONE:', out/'REPORT_ZH.md', flush=True)
    except BaseException:
        for r in queue.running.values():
            if r['process'].poll() is None:
                r['process'].terminate()      # only this controller's own children
        write(out/'QUEUE_EVENTS_AT_FAILURE.json', queue.events, replace=True)
        write(out/'FAILED.json', dict(time=utc(), traceback=traceback.format_exc(), automatic_retry=False))
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == '__main__':
    main()
