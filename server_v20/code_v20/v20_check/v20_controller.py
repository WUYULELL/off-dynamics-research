"""V20 controller: two checks chosen by the user (option C).

Part 1 (CPU, in this process): behaviour quality of source vs target trajectories on the kinematic tasks.
Part 2 (GPU): new hypothesis H_FT, tested with new seeds and whole-trajectory hold-out: a dynamics model
pretrained on source data and fine-tuned on target data (FT) beats one trained on target data only (T).
Reads the transfer project and $HOME/odrl_target_20261115.zip read-only; writes only under
$HOME/v20_check (outputs/ and data/). No simulator, no policy training.
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
PROJECT = Path(os.environ.get('V20_TEST_HOME', str(Path.home())))/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
HOME = Path(os.environ.get('V20_TEST_HOME', str(Path.home())))
ROBOTS = {'hopper': dict(spec_task='hopper-morph', state_dim=11, action_dim=3),
          'walker2d': dict(spec_task='walker2d-morph', state_dim=17, action_dim=6)}
MODELS = ('S', 'T', 'FT')
DATA = HERE/'data'
TEST = os.environ.get('V20_TEST_BACKEND')


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
        command = [sys.executable, '-u', '-B', str(HERE/'v20_worker.py'), str(folder/'job.json')]
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
    per, runs, info = {}, [], {}
    for job in jobs:
        c = read(done[job['id']]/'COMPLETED.json')
        assert c['completed'] and c['kind'] == 'task'
        r = c['result']
        assert sorted(r['metrics']) == sorted(str(f) for f in cfg['train_fractions'])
        info[(job['task'], job['seed'])] = dict(coverage=r['coverage'], segments=r['blocks'], test_segments=r['test_blocks'],
                                                test_rows=r['test_rows'], train_segments=r['train_blocks'], fits=r['fits'])
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
                if m == 'FT':
                    for other in ('T', 'S'):
                        a = [per[(tid, s, f, other)]['one_step_state'] for s in cfg['seeds']]
                        b = [per[(tid, s, f, 'FT')]['one_step_state'] for s in cfg['seeds']]
                        d, se, pos = paired(a, b)
                        row.update({'%s_minus_FT' % other: d, '%s_minus_FT_SE' % other: se, '%s_minus_FT_positive' % other: pos,
                                    'FT_better_than_%s' % other: bool(d > 0 and d > 2*se)})
                    row['FT_relative_to_T'] = row['one_step_state']/next(r['one_step_state'] for r in rows if r['task'] == tid and r['fraction'] == f and r['model'] == 'T')
                rows.append(row)
        primary = next(r for r in rows if r['task'] == tid and r['fraction'] == '1.0' and r['model'] == 'FT')
        decisions[tid] = dict(FT_better_than_T=primary['FT_better_than_T'])
    outcome = 'confirmed' if all(decisions[t]['FT_better_than_T'] for t in cfg['dense_tasks']) else 'not_confirmed'
    return rows, runs, decisions, outcome, info


def quality(cfg):
    """Part 1: returns of source and target trajectories on the kinematic tasks (CPU, descriptive)."""
    sys.path.insert(0, str(HERE))
    import numpy as np
    import v20_worker as W
    source, target = paths(cfg)
    out = {}
    for task in cfg['tasks']:
        if 'score_min' not in task:
            continue
        sd, ad = ROBOTS[task['robot']]['state_dim'], ROBOTS[task['robot']]['action_dim']
        lo, hi = task['score_min'], task['score_max']
        res = {}
        for name, path in (('source', source[task['robot']]), ('target', target[task['id']])):
            d, info = W.load(str(path), sd, ad, drop_timeouts=False)
            ends = np.flatnonzero(d['end'])
            starts = np.concatenate([[0], ends[:-1]+1])
            ret = np.array([d['rew'][a:b+1].sum() for a, b in zip(starts, ends)])
            length = ends-starts+1
            norm = 100*(ret-lo)/(hi-lo)
            full = length >= 1000
            res[name] = dict(segments=int(len(ret)), mean_length=float(length.mean()), terminal_segments=int(d['term'].sum()),
                             norm_return_mean=float(norm.mean()), norm_return_median=float(np.median(norm)),
                             norm_return_q25=float(np.percentile(norm, 25)), norm_return_q75=float(np.percentile(norm, 75)),
                             reward_per_step=float(d['rew'].mean()), full_length_segments=int(full.sum()),
                             norm_return_full_length_mean=float(norm[full].mean()) if full.any() else None)
            res[name]['_norm'] = norm
        res['source_share_above_target_mean'] = float((res['source']['_norm'] > res['target']['norm_return_mean']).mean())
        for name in ('source', 'target'):
            res[name].pop('_norm')
        res['source_minus_target_mean'] = res['source']['norm_return_mean']-res['target']['norm_return_mean']
        out[task['id']] = res
    h, w = out.get('hopper-kinematic-legjnt-medium'), out.get('walker2d-kinematic-thighjnt-medium')
    out['_expectation_met'] = bool(h and w and h['source_minus_target_mean'] > 0 and w['source_minus_target_mean'] < 0)
    return out


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


def report(out, cfg, rows, decisions, outcome, info, queue, elapsed, qual):
    names = dict(S='S（源模型直接用）', T='T（只用目标数据）', FT='FT（源模型预训练 + 目标微调）')
    text = ['# V20 两项检验：源数据与目标数据的质量对比；预训练-微调动力学模型的确认', '',
        '用户选择方案 C（2026-11-17）。第一部分只用 CPU；第二部分只比较动力学模型的预测误差，不训练策略、不运行仿真器。本次输出：`%s`。' % out, '',
        '## 预设规则（结果产生前写定）', '',
        '- 假设 H_FT 是看到 V19 后提出的（V19 中 FT 优于 T 只是描述性观察）；本次用新种子 %s、整条轨迹留出检验，不复用 V19 的结果。' % cfg['seeds'],
        '- 主指标：留出目标轨迹上的一步状态误差，训练比例 1.0；单位为该次重复全部目标训练轨迹的标准差。',
        '- “更好”：5 次重复配对，均值（T 误差 − FT 误差）> 0 且 > 2 倍配对标准误。',
        '- 判定（只看 %s）：两个任务上 FT 都优于 T → 确认；否则 → 不确认，源数据预训练不作为创新点二的依据。' % '、'.join(cfg['dense_tasks']),
        '- 第一部分为描述性检验。事先写下的预期：若源数据质量能解释 V18b，则 hopper 关节限位任务上源轨迹的归一化平均回报应高于目标轨迹，walker 大腿关节 medium 任务上应低于目标轨迹。',
        '- 模型误差更低是策略更好的必要条件，不是充分条件。规则看到结果后不修改。', '',
        '## 第一部分：源数据与目标数据的行为质量', '',
        '回报按 ODRL 参考回报归一化。源轨迹的回报是在源环境中记录的；walker 大腿关节 hard 上只有约 54% 的源轨迹能在目标环境中复现（V16），该任务的对比较弱。', '']
    qrows = []
    for task in cfg['tasks']:
        q = qual.get(task['id'])
        if not q:
            continue
        for name, lab in (('source', '源数据'), ('target', '目标数据')):
            r = q[name]
            qrows.append([task['id'], lab, r['segments'], r['mean_length'], r['norm_return_mean'], r['norm_return_median'],
                          '%s–%s' % (fmt(r['norm_return_q25']), fmt(r['norm_return_q75'])), r['reward_per_step'],
                          r['full_length_segments'], r['norm_return_full_length_mean']])
    text += [table(['任务', '数据', '轨迹段数', '平均长度', '归一化回报均值', '中位数', '四分位区间', '每步奖励', '满1000步段数', '满长段回报均值'], qrows), '']
    text += [table(['任务', '源均值 − 目标均值', '源轨迹中高于目标均值的比例'],
                   [[t, qual[t]['source_minus_target_mean'], qual[t]['source_share_above_target_mean']] for t in qual if not t.startswith('_')]), '',
             '事先写下的预期是否相符：%s。' % ('相符' if qual['_expectation_met'] else '不相符'), '',
             '## 第二部分：预训练-微调动力学模型（判定：%s）' % {'confirmed': '确认', 'not_confirmed': '不确认'}[outcome], '']
    text += [table(['任务', '类型', 'FT 优于 T'], [[t['id'], {'dense': '稠密', 'sparse': '稀疏'}[t['kind']], decisions[t['id']]['FT_better_than_T']] for t in cfg['tasks']]), '']
    for task in cfg['tasks']:
        tid = task['id']
        seg = [info[(tid, s)] for s in cfg['seeds']]
        cov = [x['coverage'] for x in seg]
        text += ['### %s（%s）' % (tid, {'dense': '稠密偏移', 'sparse': '稀疏偏移，参照'}[task['kind']]), '',
                 '目标数据共 %d 条轨迹段；每次重复留出约 %.0f 段、%.0f 行作测试。测试样本到目标训练轨迹的最近距离中位数 %.3f，到源数据（抽样）的最近距离中位数 %.3f。'
                 % (seg[0]['segments'], mean_sd([x['test_segments'] for x in seg])[0], mean_sd([x['test_rows'] for x in seg])[0],
                    mean_sd([c['test_to_train'] for c in cov])[0], mean_sd([c['test_to_source'] for c in cov])[0]), '']
        tr = [r for r in rows if r['task'] == tid and r['fraction'] == '1.0']
        text += [table(['模型', '一步状态', '近', '远', '一步奖励', '5步', '10步', '5步中位数', '10步中位数'],
                       [[names[r['model']], '%s ± %s' % (fmt(r['one_step_state']), fmt(r['one_step_state_SE'])),
                         r['one_step_state_near'], r['one_step_state_far'], r['one_step_reward'], r['rollout5_state'],
                         r['rollout10_state'], r['rollout5_state_median'], r['rollout10_state_median']] for r in tr]), '']
        ft = next(r for r in tr if r['model'] == 'FT')
        text += [table(['比较（一步状态误差）', '配对差均值', '配对SE', '正号重复数', '成立', 'FT/T'],
                       [['T − FT', ft['T_minus_FT'], ft['T_minus_FT_SE'], ft['T_minus_FT_positive'], ft['FT_better_than_T'], ft['FT_relative_to_T']],
                        ['S − FT', ft['S_minus_FT'], ft['S_minus_FT_SE'], ft['S_minus_FT_positive'], ft['FT_better_than_S'], '']]), '']
        half = {r['model']: r for r in rows if r['task'] == tid and r['fraction'] == '0.5'}
        text += ['训练比例 0.5（描述性）：T %s，FT %s，T − FT %s（SE %s）。' % (fmt(half['T']['one_step_state']), fmt(half['FT']['one_step_state']),
                 fmt(half['FT']['T_minus_FT']), fmt(half['FT']['T_minus_FT_SE'])), '']
    retries = [e for e in queue.events if e['event'] == 'start' and e['attempt'] > 0]
    text += ['## 运行情况', '',
             '总耗时 %.1f 小时，外部中断后自动重跑 %d 次；按共享规则放行 %d 次；各卡暂不放行的检查次数：%s。'
             % (elapsed/3600, len(retries), queue.shared_starts, {str(i): b['checks'] for i, b in sorted(queue.blocked.items())} or '无'),
             '本次不提出也不启动下一项实验。', '']
    (out/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    argparse.ArgumentParser().parse_args()
    for name, digest in read(HERE/'FROZEN.json')['files'].items():
        assert sha(HERE/name) == digest, ('V20 package hash mismatch', name)
    cfg = read(HERE/'config.json')
    if not TEST:
        assert sys.platform == 'linux' and sys.version_info[:3] == (3, 8, 13)
        assert Path(sys.prefix).resolve() == (Path.home()/'conda-envs/offdynamics38').resolve()
    (HERE/'outputs').mkdir(exist_ok=True)
    lock = (HERE/'outputs/v20.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = HERE/'outputs'/('check_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    (out/'runs').mkdir()
    (HERE/'outputs/LATEST_RUN.txt').write_text(str(out)+'\n', encoding='utf-8')
    print('V20 output:', out, flush=True)
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

        heartbeat('QUALITY_CHECK_CPU')
        qual = quality(cfg)
        write(out/'QUALITY.json', qual)
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
        write(out/'RESULT.json', dict(outcome=outcome, decisions=decisions, rows=rows, quality=qual,
                                      source_models={k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in source_fits.items()},
                                      task_info={'%s_s%d' % k: v for k, v in info.items()}))
        report(out, cfg, rows, decisions, outcome, info, queue, elapsed, qual)
        write(out/'COMPLETED.json', dict(completed=True, status='AWAITING_USER_REVIEW', outcome=outcome,
            elapsed_seconds=elapsed, ended_utc=utc(),
            evidence_sha256={n: sha(out/n) for n in ('REPORT_ZH.md', 'RESULT.json', 'config.json')}))
        heartbeat('STOPPED_AWAITING_USER_REVIEW')
        print('V20 DONE:', out/'REPORT_ZH.md', flush=True)
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
