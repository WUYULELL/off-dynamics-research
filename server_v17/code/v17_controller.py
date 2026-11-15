"""V17 controller: pilot B, small-data regularizers for target-only IQL and BC.

Plans the runs, starts them on GPUs that nobody else is using, retries only
externally killed runs, then audits the outputs on CPU and writes the report.
Reads the transfer project and $HOME/v13_action_fix/stage0 read-only; writes
only under $HOME/v17_reg/outputs.
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
ENVS = {'hopper-morph': ('gym.envs.mujoco.hopper_v3', 'HopperEnv'),
        'walker2d-morph': ('gym.envs.mujoco.walker2d_v3', 'Walker2dEnv')}
TEST_WORKER = os.environ.get('V17_TEST_WORKER')      # synthetic local test of the controller only


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


def protected(cfg):
    manifest = {str((PROJECT/e['path']).resolve()): e for e in read(PROJECT/'MANIFEST.json')['files']}
    spec = read(PROJECT/'cluster/paths.example.json')['tasks']
    paths = [ALGO/'offline_offline/iql_dyn.py', ALGO/'offline_offline/iql_unc.py']
    for task in cfg['tasks']:
        paths += [PROJECT/spec[task]['xml'], PROJECT/spec[task]['target_dataset']]
    hashes = {}
    for path in paths:
        entry = manifest[str(path.resolve())]
        digest = sha(path)
        assert entry['sha256'] == digest and path.stat().st_size == entry['bytes'], str(path)
        hashes[str(path)] = digest
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


def plan(cfg, out, vendor, smoke):
    spec = read(PROJECT/'cluster/paths.example.json')['tasks']
    jobs = []
    seeds = [cfg['smoke_seed']] if smoke else cfg['seeds']
    tasks = cfg['tasks'][:1] if smoke else cfg['tasks']
    for learner in ('IQL', 'BC'):
        for task in tasks:
            for seed in seeds:
                for variant in cfg['variants']:
                    steps = cfg['smoke_steps'] if smoke else cfg['steps'][learner]
                    name = '%s%s_%s_%s_s%d' % ('smoke_' if smoke else '', learner, task, variant, seed)
                    jobs.append(dict(id=name, task=task, learner=learner, variant=variant, seed=seed,
                        seed_origin=cfg['smoke_seed'] if smoke else cfg['seeds'][0], steps=steps,
                        eval_every=cfg['smoke_steps'] if smoke else cfg['eval_every'][learner],
                        weight_decay=cfg['weight_decay'], keep_fraction=cfg['keep_fraction'],
                        layer_floor_fraction=cfg['layer_floor_fraction'], mask_batch=cfg['mask_batch'],
                        mask_every=cfg['smoke_mask_every'] if smoke else cfg['mask_every'], mask_until=cfg['mask_until'],
                        state_dim=spec[task]['state_dim'], action_dim=spec[task]['action_dim'],
                        score_min=spec[task]['score_min'], score_max=spec[task]['score_max'],
                        xml=str(PROJECT/spec[task]['xml']), target_dataset=str(PROJECT/spec[task]['target_dataset']),
                        env_module=ENVS[task][0], env_class=ENVS[task][1], vendor=str(vendor), algo=str(ALGO),
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
                self.blocked[index] = self.blocked.get(index, 0)+1
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
        command = [sys.executable, '-u', '-B', TEST_WORKER or str(HERE/'v17_worker.py'), str(folder/'job.json')]
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
                               failed=len(failed), progress=progress, blocked_cards=self.blocked)
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
                wall_seconds=completed['wall_seconds'], mask_last=completed.get('mask_last'))


def summarize(cfg, jobs, done):
    table, rows = {}, []
    for job in jobs:
        r = collect(done[job['id']], job, cfg)
        table[(job['task'], job['learner'], job['variant'], job['seed'])] = r
        rows.append(dict(task=job['task'], learner=job['learner'], variant=job['variant'], seed=job['seed'], **{k: v for k, v in r.items() if k != 'mask_last'}))
    groups = []
    for task in cfg['tasks']:
        base_means = {}
        for learner in ('IQL', 'BC'):
            base = [table[(task, learner, 'base', s)]['last5'] for s in cfg['seeds']]
            base_means[learner] = stats(base)[0]
        strong = max(base_means.values())
        for learner in ('IQL', 'BC'):
            base = [table[(task, learner, 'base', s)]['last5'] for s in cfg['seeds']]
            for variant in cfg['variants']:
                values = [table[(task, learner, variant, s)]['last5'] for s in cfg['seeds']]
                mean, sd = stats(values)
                g = dict(task=task, learner=learner, variant=variant, seeds=len(values), last5_mean=mean, last5_sd=sd,
                         worst_seed=min(values), final_mean=stats([table[(task, learner, variant, s)]['final'] for s in cfg['seeds']])[0],
                         length_mean=stats([table[(task, learner, variant, s)]['length_mean'] for s in cfg['seeds']])[0],
                         fraction_under_200=stats([table[(task, learner, variant, s)]['fraction_under_200'] for s in cfg['seeds']])[0],
                         strong_baseline=strong, minus_strong_baseline=mean-strong)
                if variant != 'base':
                    diff = [a-b for a, b in zip(values, base)]
                    dmean, dsd = stats(diff)
                    se = dsd/math.sqrt(len(diff))
                    g.update(paired_difference=dmean, paired_SE=se, positive_seeds=sum(d > 0 for d in diff),
                             effective=bool(dmean > 0 and dmean > 2*se),
                             worth_confirming=bool(dmean > 0 and dmean > 2*se and mean >= strong))
                groups.append(g)
    return rows, groups


def fmt(x, digits=2):
    return '' if x is None else ('%.*f' % (digits, x) if isinstance(x, float) else str(x))


def table(header, rows):
    return '\n'.join(['|'+'|'.join(header)+'|', '|'+'|'.join(['---']*len(header))+'|']
                     + ['|'+'|'.join(fmt(c) for c in r)+'|' for r in rows])


def report(out, cfg, groups, queue, elapsed, smoke_seconds):
    text = ['# V17 试点 B：小数据正则化对只用目标数据的 IQL 与 BC 的影响', '',
        '任务：hopper-morph、walker2d-morph，只用各自 5000 条目标数据，不使用任何源数据。网络与损失为迁移项目中的冻结定义，超参数与此前的 T、BC-all 相同。',
        '本次输出：`%s`。' % out, '',
        '## 预设规则（结果产生前写定）', '',
        '- 主指标 Last5：最后五个评估点的平均（IQL 每 1 万步、BC 每 5000 步评估 10 回合）。种子 %s，同种子的各变体共用初始化和采样序列。' % cfg['seeds'],
        '- 有效：变体相对同一学习器的 base，同种子配对差的均值 > 0 且 > 2 倍标准误。',
        '- 值得确认：有效，且该变体的均值不低于该任务的强基线（base 的 IQL 与 BC 中较高者）。',
        '- 本次是 5 个种子的筛查。即使“值得确认”，也需要换新种子做确认实验后才能下结论；规则看到结果后不修改。',
        '- sparse 变体按 Sparse-Reg 论文的描述实现（保留 %g%% 的权重，每 %d 步刷新一次掩码，直到第 %d 步），论文未说明的细节由本实现自行确定，未与作者代码核对。wd 为 AdamW，系数 %g。'
        % (100*cfg['keep_fraction'], cfg['mask_every'], cfg['mask_until'], cfg['weight_decay']), '']
    for task in cfg['tasks']:
        rows = [g for g in groups if g['task'] == task]
        text += ['## %s（强基线 %.2f）' % (task, rows[0]['strong_baseline']), '',
            table(['学习器', '变体', 'Last5均值', 'SD', '最差种子', '相对base配对差', '配对SE', '正号种子', '有效', '相对强基线', '值得确认', '回合长度', '<200步比例'],
                [[g['learner'], g['variant'], g['last5_mean'], g['last5_sd'], g['worst_seed'], g.get('paired_difference'), g.get('paired_SE'),
                  g.get('positive_seeds'), {True: '是', False: '否', None: ''}[g.get('effective')], g['minus_strong_baseline'],
                  {True: '是', False: '否', None: ''}[g.get('worth_confirming')], g['length_mean'], g['fraction_under_200']] for g in rows]), '']
    retries = [e for e in queue.events if e['event'] == 'start' and e['attempt'] > 0]
    text += ['## 运行情况', '',
        '正式训练 %d 项，总耗时 %.1f 小时（另有冒烟测试 %.0f 秒，其结果不计入）。外部中断后自动重跑 %d 次。'
        % (len([e for e in queue.events if e['event'] == 'exit' and e['completed'] and not e['job'].startswith('smoke_')]), elapsed/3600, smoke_seconds, len(retries)),
        '因检测到他人进程或占用而多次（≥10 次检查）暂不放行的卡：%s。' % ([i for i, c in sorted(queue.blocked.items()) if c >= 10] or '无'),
        '与此前结果的关系：本次 base 使用新的种子和新的随机流，数值只与本次的变体配对比较；与 v12/v13 的 T、BC-all 只作描述性对照。',
        '本次不提出也不启动下一项实验。', '']
    (out/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    for name, digest in read(HERE/'FROZEN.json')['files'].items():
        assert sha(HERE/name) == digest, ('V17 package hash mismatch', name)
    cfg = read(HERE/'config.json')
    if not TEST_WORKER:
        assert sys.platform == 'linux' and sys.version_info[:3] == (3, 8, 13)
        assert Path(sys.prefix).resolve() == (Path.home()/'conda-envs/offdynamics38').resolve()
    (HERE/'outputs').mkdir(exist_ok=True)
    lock = (HERE/'outputs/v17.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = HERE/'outputs'/('pilotB_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    (out/'runs').mkdir()
    (HERE/'outputs/LATEST_RUN.txt').write_text(str(out)+'\n', encoding='utf-8')
    print('V17 output:', out, flush=True)
    started = time.monotonic()

    def heartbeat(phase, **kw):
        write(out/'STATUS.json', dict(time=utc(), phase=phase, **kw), replace=True)
        print(phase, {k: v for k, v in kw.items() if k != 'progress'}, flush=True)

    queue = Queue(cfg, out, heartbeat)
    try:
        heartbeat('VERIFYING_INPUTS')
        before = protected(cfg)
        write(out/'PROTECTED_BEFORE.json', before)
        write(out/'config.json', cfg)
        write(out/'GPU_AT_START.json', gpus())
        if TEST_WORKER:
            vendor = out
        else:
            sys.path.insert(0, str(V13/'stage0'))
            import importlib.util
            spec = importlib.util.spec_from_file_location('v17_binding_loader', str(V13/'stage0/legacy_replay.py'))
            legacy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(legacy)
            binding, binding_before, private = legacy.prepare_private_binding(out)
            vendor = private.parent
            sys.path.insert(0, str(vendor))
            import mujoco_py                    # build the private copy once before any worker starts
            assert private.resolve() in Path(mujoco_py.cymj.__file__).resolve().parents
        smoke_jobs = plan(cfg, out, vendor, smoke=True)
        heartbeat('SMOKE_TEST', jobs=len(smoke_jobs))
        t0 = time.monotonic()
        done, failed = queue.run(smoke_jobs, 'SMOKE_TEST')
        smoke_seconds = time.monotonic()-t0
        if failed or len(done) != len(smoke_jobs):
            raise RuntimeError('Smoke test failed: %s' % json.dumps(failed, ensure_ascii=False))
        for job in smoke_jobs:
            completed = read(done[job['id']]/'COMPLETED.json')
            assert completed['steps'] == cfg['smoke_steps'] and completed['finite'] and completed['evaluations'] == 1
            if job['variant'] == 'sparse':
                assert completed['mask_refreshes'] >= 2, 'mask refresh did not run in the smoke test'
                density = [d for values in completed['mask_last'][0]['density'].values() for d in values]
                assert all(0 < d <= 1 for d in density) and min(density) < 1, 'mask density out of range'
        write(out/'SMOKE_PASSED.json', dict(passed=True, jobs=len(smoke_jobs), seconds=smoke_seconds,
            note='Smoke runs use a separate seed and %d steps; their returns are not read or reported.' % cfg['smoke_steps']))
        jobs = plan(cfg, out, vendor, smoke=False)
        write(out/'PLAN.json', dict(jobs=[j['id'] for j in jobs], count=len(jobs)))
        heartbeat('TRAINING', jobs=len(jobs))
        done, failed = queue.run(jobs, 'TRAINING')
        write(out/'QUEUE_EVENTS.json', queue.events)
        if failed or len(done) != len(jobs):
            raise RuntimeError('Training runs failed: %s' % json.dumps(failed, ensure_ascii=False))
        heartbeat('AUDITING')
        rows, groups = summarize(cfg, jobs, done)
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
        report(out, cfg, groups, queue, elapsed, smoke_seconds)
        write(out/'COMPLETED.json', dict(completed=True, status='AWAITING_USER_REVIEW', runs=len(rows),
            worth_confirming=[(g['task'], g['learner'], g['variant']) for g in groups if g.get('worth_confirming')],
            elapsed_seconds=elapsed, ended_utc=utc(),
            evidence_sha256={n: sha(out/n) for n in ('REPORT_ZH.md', 'RESULT.json', 'config.json')}))
        heartbeat('STOPPED_AWAITING_USER_REVIEW')
        print('V17 DONE:', out/'REPORT_ZH.md', flush=True)
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
