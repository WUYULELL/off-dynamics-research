"""V19 controller: residual dynamics-model pilot.

Phase 1 trains one source dynamics model per robot and seed; phase 2 runs one job per task and
seed (all train fractions, models T, FT, RES, and S evaluated). Starts jobs on GPUs under the same
admission rule as V18b, retries only externally killed jobs, audits, writes the report.
Reads the transfer project and $HOME/odrl_target_20261115.zip read-only; writes only under
$HOME/v19_residual (outputs/ and data/). No simulator, no policy training.
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
PROJECT = Path(os.environ.get('V19_TEST_HOME', str(Path.home())))/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
HOME = Path(os.environ.get('V19_TEST_HOME', str(Path.home())))
ROBOTS = {'hopper': dict(spec_task='hopper-morph', state_dim=11, action_dim=3),
          'walker2d': dict(spec_task='walker2d-morph', state_dim=17, action_dim=6)}
MODELS = ('S', 'T', 'FT', 'RES')
DATA = HERE/'data'
TEST = os.environ.get('V19_TEST_BACKEND')


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
    """Unpack the ODRL target files once into this package's own data folder and verify the ones used."""
    import zipfile
    archive = HOME/cfg['target_archive']
    if not DATA.is_dir():
        assert archive.is_file(), 'Upload %s to the home directory first' % cfg['target_archive']
        assert sha(archive) == cfg['target_archive_sha256'], 'Target archive hash mismatch'
        tmp = HERE/('data.tmp.'+uuid.uuid4().hex)
        tmp.mkdir()
        with zipfile.ZipFile(str(archive)) as z:
            for info in z.infolist():
                if info.filename.endswith('.hdf5') and '/raw/' in info.filename:
                    (tmp/Path(info.filename).name).write_bytes(z.read(info))
        os.replace(str(tmp), str(DATA))
    for task in cfg['tasks']:
        if task['origin'] == 'odrl_archive':
            assert sha(DATA/task['target_file']) == task['target_sha256'], task['target_file']


def paths(cfg):
    """Source dataset per robot and target dataset per task."""
    spec = read(PROJECT/'cluster/paths.example.json')['tasks']
    source = {r: PROJECT/spec[ROBOTS[r]['spec_task']]['source_dataset'] for r in sorted({t['robot'] for t in cfg['tasks']})}
    target = {}
    for task in cfg['tasks']:
        target[task['id']] = PROJECT/spec[task['spec_task']]['target_dataset'] if task['origin'] == 'transfer_project' else DATA/task['target_file']
    return source, target


def protected(cfg):
    manifest = {str((PROJECT/e['path']).resolve()): e for e in read(PROJECT/'MANIFEST.json')['files']}
    source, target = paths(cfg)
    hashes = {}
    for path in list(source.values())+[target[t['id']] for t in cfg['tasks'] if t['origin'] == 'transfer_project']:
        entry = manifest[str(path.resolve())]
        digest = sha(path)
        assert entry['sha256'] == digest and path.stat().st_size == entry['bytes'], str(path)
        hashes[str(path)] = digest
    for task in cfg['tasks']:
        hashes.setdefault(str(target[task['id']]), sha(target[task['id']]))
    return hashes


def smi(*args):
    return subprocess.run(['nvidia-smi']+list(args)+['--format=csv,noheader,nounits'], check=True,
                          stdout=subprocess.PIPE, universal_newlines=True, timeout=60).stdout.strip().splitlines()


def gpus():
    if TEST:
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


def plan_source(cfg, out, hashes, smoke):
    source, _ = paths(cfg)
    seeds = [cfg['smoke_seed']] if smoke else cfg['seeds']
    robots = [next(t['robot'] for t in cfg['tasks'] if t['id'] == cfg['smoke']['task'])] if smoke else sorted(source)
    train = dict(cfg['source_training'])
    if smoke:
        train.update(steps=cfg['smoke']['source_steps'], eval_every=cfg['smoke']['source_eval_every'])
    jobs = []
    for robot in robots:
        for seed in seeds:
            jobs.append(dict(id='%ssource_%s_s%d' % ('smoke_' if smoke else '', robot, seed), kind='source', robot=robot,
                             seed=seed, state_dim=ROBOTS[robot]['state_dim'], action_dim=ROBOTS[robot]['action_dim'],
                             source_dataset=str(source[robot]), source_dataset_sha256=hashes[str(source[robot])],
                             network=cfg['network'], train=train, total_steps=train['steps'], smoke=smoke))
    return jobs


def plan_tasks(cfg, out, hashes, source_done, smoke):
    source, target = paths(cfg)
    seeds = [cfg['smoke_seed']] if smoke else cfg['seeds']
    tasks = [t for t in cfg['tasks'] if t['id'] == cfg['smoke']['task']] if smoke else cfg['tasks']
    train = dict(cfg['target_training'])
    if smoke:
        train.update(max_steps=cfg['smoke']['target_max_steps'], eval_every=cfg['smoke']['target_eval_every'])
    jobs = []
    for task in tasks:
        robot = task['robot']
        for seed in seeds:
            src_id = '%ssource_%s_s%d' % ('smoke_' if smoke else '', robot, seed)
            jobs.append(dict(id='%stask_%s_s%d' % ('smoke_' if smoke else '', task['id'], seed), kind='task', task=task['id'],
                             robot=robot, task_kind=task['kind'], seed=seed,
                             state_dim=ROBOTS[robot]['state_dim'], action_dim=ROBOTS[robot]['action_dim'],
                             source_dataset=str(source[robot]), source_dataset_sha256=hashes[str(source[robot])],
                             target_dataset=str(target[task['id']]), target_dataset_sha256=hashes[str(target[task['id']])],
                             source_model_dir=str(source_done[src_id]), network=cfg['network'], train=train,
                             split=cfg['split'], fractions=cfg['smoke']['fractions'] if smoke else cfg['train_fractions'],
                             horizons=cfg['rollout_horizons'], coverage_sample=cfg['source_coverage_sample'], smoke=smoke))
    return jobs


class Queue:
    """Same GPU admission rule as V18b: free cards first; a card someone else uses is joined only while the
    whole card is at most shared_utilization_max %% busy with shared_free_memory_min_MiB free, one start per cooldown."""
    def __init__(self, cfg, out, heartbeat):
        self.cfg, self.out, self.heartbeat = cfg, out, heartbeat
        self.running, self.events, self.blocked = {}, [], {}
        self.last_start, self.shared_starts = {}, 0

    def card(self):
        state = gpus()
        exclusive, shared = None, None
        for index, row in sorted(state.items()):
            if index not in self.cfg['gpu_indices']:
                continue
            mine = sum(1 for r in self.running.values() if r['gpu'] == index)
            if mine >= self.cfg['max_per_gpu']:
                continue
            foreign = row['processes'] > mine or (mine == 0 and (row['utilization'] > self.cfg['idle_utilization_max']
                                                               or row['used_MiB'] > self.cfg['idle_memory_max_MiB']))
            if not foreign:
                if exclusive is None or mine < exclusive[1]:
                    exclusive = (index, mine)
                continue
            free = row['total_MiB']-row['used_MiB']
            ok = (row['utilization'] <= self.cfg['shared_utilization_max'] and free >= self.cfg['shared_free_memory_min_MiB']
                  and time.time()-self.last_start.get(index, 0) >= self.cfg['shared_cooldown_seconds'])
            if ok:
                if shared is None or row['utilization'] < shared[1]:
                    shared = (index, row['utilization'], dict(row, own_runs=mine))
                continue
            b = self.blocked.setdefault(index, dict(checks=0, samples=[]))
            b['checks'] += 1
            if len(b['samples']) < 20 and (not b['samples'] or time.time()-b['samples'][-1]['t'] > 60):
                b['samples'].append(dict(t=time.time(), time=utc(), own_runs=mine, driver_processes=row['processes'],
                                         utilization=row['utilization'], used_MiB=row['used_MiB']))
        if exclusive is not None:
            return exclusive[0]
        if shared is not None:
            self.shared_starts += 1
            self.events.append(dict(time=utc(), event='shared_admission', gpu=shared[0], reading=shared[2]))
            return shared[0]
        return None

    def start(self, job, gpu, attempt):
        folder = self.out/'runs'/(job['id'] if attempt == 0 else '%s_retry%d' % (job['id'], attempt))
        folder.mkdir(parents=True)
        job = dict(job, gpu=gpu, output=str(folder), attempt=attempt)
        write(folder/'job.json', job)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        command = [sys.executable, '-u', '-B', str(HERE/'v19_worker.py'), str(folder/'job.json')]
        log = (folder/'stdout.log').open('w')
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(folder))
        self.last_start[gpu] = time.time()
        self.running[job['id']] = dict(job=job, process=process, gpu=gpu, folder=folder, log=log, attempt=attempt)
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
                    pending.insert(0, dict(r['job'], _attempt=r['attempt']+1))
                else:
                    failed[name] = dict(folder=str(r['folder']), returncode=code, attempt=r['attempt'])
            if failed:
                pending = []
            while pending:
                gpu = self.card()
                if gpu is None:
                    break
                job = pending.pop(0)
                attempt = job.pop('_attempt', 0)
                for k in ('gpu', 'output', 'attempt'):
                    job.pop(k, None)
                self.start(job, gpu, attempt)
                time.sleep(0 if TEST else self.cfg['launch_gap_seconds'])
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
            time.sleep(0.2 if TEST else 5)
        return done, failed


def mean_sd(x):
    n = len(x)
    m = sum(x)/n
    return m, (math.sqrt(sum((v-m)**2 for v in x)/(n-1)) if n > 1 else 0.)


def paired(a, b):
    """Mean and SE of a - b over repetitions."""
    diff = [x-y for x, y in zip(a, b)]
    m, sd = mean_sd(diff)
    return m, sd/math.sqrt(len(diff)), sum(d > 0 for d in diff)


METRICS = ('one_step_state', 'one_step_state_near', 'one_step_state_far', 'one_step_reward',
           'rollout5_state', 'rollout10_state', 'rollout5_state_median', 'rollout10_state_median')


def summarize(cfg, jobs, done):
    per = {}
    runs = []
    info = {}
    for job in jobs:
        c = read(done[job['id']]/'COMPLETED.json')
        assert c['completed'] and c['kind'] == 'task'
        r = c['result']
        assert sorted(r['metrics']) == sorted(str(f) for f in cfg['train_fractions'])
        info[(job['task'], job['seed'])] = dict(coverage=r['coverage'], blocks=r['blocks'], test_rows=r['test_rows'],
                                                train_blocks=r['train_blocks'], fits=r['fits'])
        for f, models in r['metrics'].items():
            assert sorted(models) == sorted(MODELS)
            for m, vals in models.items():
                assert all(math.isfinite(v) for v in vals.values())
                per[(job['task'], job['seed'], f, m)] = vals
                runs.append(dict(task=job['task'], seed=job['seed'], fraction=f, model=m, **{k: vals[k] for k in METRICS}))
    rows, decisions = [], {}
    for task in cfg['tasks']:
        tid = task['id']
        for f in [str(x) for x in cfg['train_fractions']]:
            for m in MODELS:
                row = dict(task=tid, kind=task['kind'], fraction=f, model=m)
                for k in METRICS:
                    vals = [per[(tid, s, f, m)][k] for s in cfg['seeds']]
                    mu, sd = mean_sd(vals)
                    row[k] = mu
                    row[k+'_SE'] = sd/math.sqrt(len(vals))
                for other in ('S', 'T', 'FT'):
                    if other != m and m == 'RES':
                        a = [per[(tid, s, f, other)]['one_step_state'] for s in cfg['seeds']]
                        b = [per[(tid, s, f, 'RES')]['one_step_state'] for s in cfg['seeds']]
                        d, se, pos = paired(a, b)
                        row.update({'%s_minus_RES' % other: d, '%s_minus_RES_SE' % other: se, '%s_minus_RES_positive' % other: pos,
                                    'RES_better_than_%s' % other: bool(d > 0 and d > 2*se)})
                if m == 'FT':
                    a = [per[(tid, s, f, 'T')]['one_step_state'] for s in cfg['seeds']]
                    b = [per[(tid, s, f, 'FT')]['one_step_state'] for s in cfg['seeds']]
                    d, se, pos = paired(a, b)
                    row.update(T_minus_FT=d, T_minus_FT_SE=se, FT_better_than_T=bool(d > 0 and d > 2*se))
                rows.append(row)
        primary = next(r for r in rows if r['task'] == tid and r['fraction'] == '1.0' and r['model'] == 'RES')
        decisions[tid] = dict(RES_better_than_T=primary['RES_better_than_T'], RES_better_than_FT=primary['RES_better_than_FT'])
    dense = cfg['dense_tasks']
    if all(decisions[t]['RES_better_than_T'] and decisions[t]['RES_better_than_FT'] for t in dense):
        outcome = 'A'
    elif all(decisions[t]['RES_better_than_T'] for t in dense):
        outcome = 'B'
    else:
        outcome = 'C'
    return rows, runs, decisions, outcome, info


def fmt(x, digits=3):
    if x is None:
        return ''
    if isinstance(x, bool):
        return '是' if x else '否'
    if isinstance(x, float):
        return '%.*g' % (digits+1, x) if abs(x) < 1e-2 or abs(x) >= 1e4 else '%.*f' % (digits, x)
    return str(x)


def table(header, rows):
    return '\n'.join(['|'+'|'.join(header)+'|', '|'+'|'.join(['---']*len(header))+'|']
                     + ['|'+'|'.join(fmt(c) for c in r)+'|' for r in rows])


def report(out, cfg, rows, decisions, outcome, info, queue, elapsed):
    names = dict(S='S（源模型直接用）', T='T（只用目标数据）', FT='FT（源模型微调）', RES='RES（源模型 + 残差）')
    text = ['# V19 残差动力学模型试点：源数据能否帮助预测目标环境的动力学', '',
        '只比较动力学模型在留出目标数据上的预测误差，不训练策略、不运行仿真器。本次输出：`%s`。' % out, '',
        '## 预设规则（结果产生前写定）', '',
        '- 误差单位：该次重复中全部目标训练块的标准差（一步误差按状态变化量，多步误差按状态），各维平方误差取平均。',
        '- 主指标：留出测试块上的一步状态误差，训练比例 1.0。',
        '- “更好”：5 次重复（同一划分、同一种子）配对，均值（其他模型误差 − RES 误差）> 0 且 > 2 倍配对标准误。',
        '- 判定（只看两个稠密偏移任务 %s）：' % '、'.join(cfg['dense_tasks']),
        '  - A：两个任务上 RES 都优于 T 和 FT → 支持残差方向，下一步可做“在模型生成的数据上训练策略”的试点（本包不启动）。',
        '  - B：两个任务上 RES 都优于 T，但至少一个任务不优于 FT → 源数据的动力学知识有用，但残差结构相对微调没有优势；创新点二写成“源数据预训练的目标动力学模型”，残差只是一种实现。',
        '  - C：其他情况 → 稠密偏移下源数据对预测目标动力学没有可靠帮助，残差方向不进入策略试点。',
        '- 稀疏偏移（关节限位）任务只作参照，不参与判定。多步误差、远近分组、奖励误差、训练比例扫描均为描述性结果。',
        '- 模型误差更低是策略更好的必要条件，不是充分条件。规则看到结果后不修改。', '',
        '## 判定结果：%s' % outcome, '']
    text += [table(['任务', '类型', 'RES 优于 T', 'RES 优于 FT'],
                   [[t['id'], {'dense': '稠密', 'sparse': '稀疏'}[t['kind']], decisions[t['id']]['RES_better_than_T'],
                     decisions[t['id']]['RES_better_than_FT']] for t in cfg['tasks']]), '']
    for task in cfg['tasks']:
        tid = task['id']
        cov = [info[(tid, s)]['coverage'] for s in cfg['seeds']]
        text += ['## %s（%s）' % (tid, {'dense': '稠密偏移', 'sparse': '稀疏偏移，参照'}[task['kind']]), '',
                 '目标数据切成 %d 个块，测试块约 %d 行。测试样本到目标训练块的最近距离中位数 %.3f，到源数据（抽样 %d 行）的最近距离中位数 %.3f（标准化后的状态-动作空间，5 次重复平均）。'
                 % (info[(tid, cfg['seeds'][0])]['blocks'], info[(tid, cfg['seeds'][0])]['test_rows'],
                    mean_sd([c['test_to_train'] for c in cov])[0], cfg['source_coverage_sample'], mean_sd([c['test_to_source'] for c in cov])[0]), '',
                 '训练比例 1.0（均值 ± 标准误，5 次重复）：', '']
        tr = [r for r in rows if r['task'] == tid and r['fraction'] == '1.0']
        text += [table(['模型', '一步状态', '近', '远', '一步奖励', '5步', '10步', '5步中位数', '10步中位数'],
                       [[names[r['model']], '%s ± %s' % (fmt(r['one_step_state']), fmt(r['one_step_state_SE'])),
                         r['one_step_state_near'], r['one_step_state_far'], r['one_step_reward'], r['rollout5_state'],
                         r['rollout10_state'], r['rollout5_state_median'], r['rollout10_state_median']] for r in tr]), '']
        res = next(r for r in tr if r['model'] == 'RES')
        ft = next(r for r in tr if r['model'] == 'FT')
        text += [table(['比较（一步状态误差）', '配对差均值', '配对SE', '正号重复数', '成立'],
                       [['T − RES', res['T_minus_RES'], res['T_minus_RES_SE'], res['T_minus_RES_positive'], res['RES_better_than_T']],
                        ['FT − RES', res['FT_minus_RES'], res['FT_minus_RES_SE'], res['FT_minus_RES_positive'], res['RES_better_than_FT']],
                        ['S − RES', res['S_minus_RES'], res['S_minus_RES_SE'], res['S_minus_RES_positive'], res['RES_better_than_S']],
                        ['T − FT', ft['T_minus_FT'], ft['T_minus_FT_SE'], '', ft['FT_better_than_T']]]), '',
                 '训练比例扫描（一步状态误差均值；S 不随比例变化）：', '']
        sweep = []
        for f in [str(x) for x in cfg['train_fractions']]:
            by = {r['model']: r for r in rows if r['task'] == tid and r['fraction'] == f}
            sweep.append([f, info[(tid, cfg['seeds'][0])]['train_blocks'][f], by['S']['one_step_state'], by['T']['one_step_state'],
                          by['FT']['one_step_state'], by['RES']['one_step_state'], by['RES']['T_minus_RES'], by['RES']['RES_better_than_T']])
        text += [table(['训练比例', '训练块数', 'S', 'T', 'FT', 'RES', 'T − RES', 'RES 优于 T'], sweep), '']
    retries = [e for e in queue.events if e['event'] == 'start' and e['attempt'] > 0]
    text += ['## 运行情况', '',
             '总耗时 %.1f 小时，外部中断后自动重跑 %d 次；按共享规则放行 %d 次；各卡暂不放行的检查次数：%s。'
             % (elapsed/3600, len(retries), queue.shared_starts, {str(i): b['checks'] for i, b in sorted(queue.blocked.items())} or '无'),
             '本次不提出也不启动下一项实验。', '']
    (out/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    argparse.ArgumentParser().parse_args()
    for name, digest in read(HERE/'FROZEN.json')['files'].items():
        assert sha(HERE/name) == digest, ('V19 package hash mismatch', name)
    cfg = read(HERE/'config.json')
    if not TEST:
        assert sys.platform == 'linux' and sys.version_info[:3] == (3, 8, 13)
        assert Path(sys.prefix).resolve() == (Path.home()/'conda-envs/offdynamics38').resolve()
    (HERE/'outputs').mkdir(exist_ok=True)
    lock = (HERE/'outputs/v19.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = HERE/'outputs'/('residual_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    (out/'runs').mkdir()
    (HERE/'outputs/LATEST_RUN.txt').write_text(str(out)+'\n', encoding='utf-8')
    print('V19 output:', out, flush=True)
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

        smoke_src = plan_source(cfg, out, before, True)
        heartbeat('SMOKE_SOURCE', jobs=len(smoke_src))
        done_src, failed = queue.run(smoke_src, 'SMOKE_SOURCE')
        if failed or len(done_src) != len(smoke_src):
            raise RuntimeError('Smoke test (source) failed: %s' % json.dumps(failed, ensure_ascii=False))
        smoke_task = plan_tasks(cfg, out, before, done_src, True)
        heartbeat('SMOKE_TASK', jobs=len(smoke_task))
        done, failed = queue.run(smoke_task, 'SMOKE_TASK')
        if failed or len(done) != len(smoke_task):
            raise RuntimeError('Smoke test (task) failed: %s' % json.dumps(failed, ensure_ascii=False))
        write(out/'SMOKE_PASSED.json', dict(passed=True, note='Smoke runs use seed %d and few steps; their errors are not read or reported.' % cfg['smoke_seed']))

        src_jobs = plan_source(cfg, out, before, False)
        heartbeat('SOURCE_MODELS', jobs=len(src_jobs))
        done_src, failed = queue.run(src_jobs, 'SOURCE_MODELS')
        if failed or len(done_src) != len(src_jobs):
            raise RuntimeError('Source models failed: %s' % json.dumps(failed, ensure_ascii=False))
        task_jobs = plan_tasks(cfg, out, before, done_src, False)
        write(out/'PLAN.json', dict(source=[j['id'] for j in src_jobs], tasks=[j['id'] for j in task_jobs]))
        heartbeat('TASKS', jobs=len(task_jobs))
        done, failed = queue.run(task_jobs, 'TASKS')
        write(out/'QUEUE_EVENTS.json', queue.events)
        write(out/'GPU_BLOCKED.json', {str(k): v for k, v in queue.blocked.items()})
        if failed or len(done) != len(task_jobs):
            raise RuntimeError('Task jobs failed: %s' % json.dumps(failed, ensure_ascii=False))

        heartbeat('AUDITING')
        rows, runs, decisions, outcome, info = summarize(cfg, task_jobs, done)
        assert protected(cfg) == before, 'Protected original inputs changed'
        source_fits = {j['id']: read(done_src[j['id']]/'COMPLETED.json')['result']['fit'] for j in src_jobs}
        elapsed = time.monotonic()-started
        import csv
        with (out/'RUNS.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(runs[0]))
            w.writeheader()
            w.writerows(runs)
        write(out/'RESULT.json', dict(outcome=outcome, decisions=decisions, rows=rows,
                                      source_models={k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in source_fits.items()},
                                      task_info={'%s_s%d' % k: v for k, v in info.items()}))
        report(out, cfg, rows, decisions, outcome, info, queue, elapsed)
        write(out/'COMPLETED.json', dict(completed=True, status='AWAITING_USER_REVIEW', outcome=outcome,
            elapsed_seconds=elapsed, ended_utc=utc(),
            evidence_sha256={n: sha(out/n) for n in ('REPORT_ZH.md', 'RESULT.json', 'config.json')}))
        heartbeat('STOPPED_AWAITING_USER_REVIEW')
        print('V19 DONE:', out/'REPORT_ZH.md', flush=True)
    except BaseException:
        for r in queue.running.values():
            if r['process'].poll() is None:
                r['process'].terminate()
        write(out/'QUEUE_EVENTS_AT_FAILURE.json', queue.events, replace=True)
        write(out/'FAILED.json', dict(time=utc(), traceback=traceback.format_exc(), automatic_retry=False))
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == '__main__':
    main()
