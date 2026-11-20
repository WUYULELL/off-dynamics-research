"""j008 controller (one GPU, three tasks): smoke runs, source dynamics ensemble, 60 BC-mix runs, model-based weight
selection per (task, seed), summary and report with the registered verdict.

Uses only the GPU the relay put in CUDA_VISIBLE_DEVICES, at most max_per_gpu worker processes on it. Reads the transfer
project, the ODRL archive and the installed mujoco_py read-only; writes only under this job's code/ folder.
Failure policy (as V21/j004-j007): a worker killed by an outside signal is rerun from step 0 with the same seed in a new
folder (at most 3 times); a code error / NaN stops new launches, running workers finish, exit code 1.
Rerun decisions never depend on returns or model scores.
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
LOCAL_TEST = os.environ.get('J008_LOCAL_TEST')          # synthetic local test of the controller with fake workers
ROBOTS = {'hopper': dict(env=('gym.envs.mujoco.hopper_v3', 'HopperEnv'), source_task='hopper-morph', sd=11, ad=3),
          'walker2d': dict(env=('gym.envs.mujoco.walker2d_v3', 'Walker2dEnv'), source_task='walker2d-morph', sd=17, ad=6)}
ORDER = ('W0', 'W25', 'W50', 'W100')
NAMES = {'W0': 'BC（w=0，只用目标数据）', 'W25': 'W25（源权重 0.25）', 'W50': 'W50（源权重 0.5）',
         'W100': 'BC-混合（w=1）', 'SEL': 'SEL（候选：模型选权重）'}
SHORT = {'hopper-kinematic-legjnt-medium': 'hopper medium', 'walker2d-kinematic-thighjnt-medium': 'walker medium',
         'walker2d-kinematic-thighjnt-hard': 'walker hard'}


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
    """Frozen hashes, private mujoco_py, target data, source/learner hashes."""
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
    p = ALGO/'offline_offline/iql_unc.py'
    assert sha(p) == files[str(p.resolve())]['sha256'], str(p)
    for robot, r in ROBOTS.items():
        p = PROJECT/paths_spec[r['source_task']]['source_dataset']
        d = sha(p)
        assert d == files[str(p.resolve())]['sha256'], str(p)
        sources[robot], source_sha[robot] = str(p), d
    return str(vendor), str(ALGO), data, sources, source_sha


def task_common(cfg, task, vendor, algo, data, sources, source_sha):
    robot = ROBOTS[task['robot']]
    return dict(task=task['id'], robot=task['robot'], state_dim=robot['sd'], action_dim=robot['ad'],
                target_rows=cfg['target_rows'], score_min=task['score_min'], score_max=task['score_max'],
                xml=str(HERE/'assets'/task['xml']), xml_sha256=task['xml_sha256'],
                target_dataset=str(data/task['target_file']), target_dataset_sha256=task['target_sha256'],
                source_dataset=sources[task['robot']], source_dataset_sha256=source_sha[task['robot']],
                env_module=robot['env'][0], env_class=robot['env'][1], vendor=vendor, algo=algo)


def plan_stage1(cfg, vendor, algo, data, sources, source_sha, smoke):
    """Source ensemble members (first: they are short and needed later) and the policy runs."""
    jobs = []
    pre = 'smoke_' if smoke else ''
    for robot in sorted({t['robot'] for t in cfg['tasks']}):
        r = ROBOTS[robot]
        for k in range(cfg['model']['members']):
            jobs.append(dict(id='%ssource_%s_m%d' % (pre, robot, k), script='model_worker.py', kind='source', robot=robot, member=k,
                             state_dim=r['sd'], action_dim=r['ad'], model=cfg['model'], source_dataset=sources[robot],
                             source_dataset_sha256=source_sha[robot],
                             source_steps=cfg['smoke']['source_steps'] if smoke else cfg['model']['source_training']['steps'],
                             source_eval_every=cfg['smoke']['source_eval_every'] if smoke else cfg['model']['source_training']['eval_every'],
                             smoke=smoke))
    seeds = [cfg['smoke']['seed']] if smoke else cfg['seeds']
    for group in ORDER:
        for task in cfg['tasks']:
            for seed in seeds:
                job = dict(task_common(cfg, task, vendor, algo, data, sources, source_sha),
                           id='%s%s_%s_s%d' % (pre, group, task['id'], seed), script='worker.py', kind='policy', group=group,
                           seed=seed, seed_origin=seeds[0], steps=cfg['smoke']['policy_steps'] if smoke else cfg['steps']['BC'],
                           eval_every=cfg['smoke']['policy_steps'] if smoke else cfg['eval_every']['BC'], smoke=smoke)
                jobs.append(job)
    if LOCAL_TEST:
        for j in jobs:
            j.update(read(LOCAL_TEST)['job_overrides'])
    return jobs


def plan_select(cfg, stage1, done, vendor, algo, data, sources, source_sha, smoke):
    jobs = []
    pre = 'smoke_' if smoke else ''
    seeds = [cfg['smoke']['seed']] if smoke else cfg['seeds']
    by_id = {j['id']: j for j in stage1}
    for task in cfg['tasks']:
        for seed in seeds:
            source_dirs = [str(done['%ssource_%s_m%d' % (pre, task['robot'], k)]) for k in range(cfg['model']['members'])]
            policy_dirs = {g: str(done['%s%s_%s_s%d' % (pre, g, task['id'], seed)]) for g in ORDER}
            steps = by_id['%sW0_%s_s%d' % (pre, task['id'], seed)]['steps']
            jobs.append(dict(task_common(cfg, task, vendor, algo, data, sources, source_sha),
                             id='%sselect_%s_s%d' % (pre, task['id'], seed), script='model_worker.py', kind='select', seed=seed,
                             model=cfg['model'], rollout=cfg['rollout'], source_dirs=source_dirs, policy_dirs=policy_dirs,
                             policy_steps=steps,
                             target_max_steps=cfg['smoke']['target_max_steps'] if smoke else cfg['model']['target_training']['max_steps'],
                             target_eval_every=cfg['smoke']['target_eval_every'] if smoke else cfg['model']['target_training']['eval_every'],
                             horizon=cfg['smoke']['horizon'] if smoke else cfg['rollout']['horizon'],
                             short_horizon=cfg['smoke']['short_horizon'] if smoke else cfg['rollout']['short_horizon'],
                             smoke=smoke))
    if LOCAL_TEST:
        for j in jobs:
            j.update(read(LOCAL_TEST)['job_overrides'])
    return jobs


def visible_gpus():
    value = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    gpus = [g.strip() for g in value.split(',') if g.strip() and g.strip() != '-1']
    assert gpus, 'CUDA_VISIBLE_DEVICES is empty or -1: this job needs the GPU the relay assigns'
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
            worker = read(LOCAL_TEST)['workers'][job['script']] if LOCAL_TEST else str(HERE/job['script'])
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
        time.sleep(0.2 if LOCAL_TEST else 10)
    write(OUT/('events_%s.json' % phase), events)
    return done, failed


# ----------------------------------------------------------------------------- statistics

def stats(x):
    n = len(x)
    mean = sum(x)/n
    sd = math.sqrt(sum((v-mean)**2 for v in x)/(n-1)) if n > 1 else 0.
    return mean, sd, sd/math.sqrt(n)


def paired(a, b):
    diff = [x-y for x, y in zip(a, b)]
    mean, sd, se = stats(diff)
    return dict(mean=mean, sd=sd, se=se, positive=sum(d > 0 for d in diff), diffs=diff)


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.]*len(x)
    i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and x[order[j+1]] == x[order[i]]:
            j += 1
        for k in range(i, j+1):
            r[order[k]] = (i+j)/2.+1
        i = j+1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra)/len(ra), sum(rb)/len(rb)
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb))
    return num/den if den > 0 else 0.


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
                length_mean=sum(lengths)/len(lengths), fraction_under_200=sum(l < 200 for l in lengths)/len(lengths),
                curve=[round(e['mean'], 3) for e in evals], wall_seconds=completed['wall_seconds'])


def summarize(cfg, policy_jobs, done, select_jobs, sel_done):
    table, runs = {}, []
    for job in policy_jobs:
        r = collect(done[job['id']], job)
        table[(job['task'], job['group'], job['seed'])] = r
        runs.append(dict(task=job['task'], group=job['group'], seed=job['seed'], **{k: v for k, v in r.items() if k != 'curve'}))
    choice = {}
    for job in select_jobs:
        res = read(sel_done[job['id']]/'COMPLETED.json')['result']
        assert res['task'] == job['task'] and res['seed'] == job['seed'] and res['chosen'] in ORDER
        choice[(job['task'], job['seed'])] = res
    seeds = cfg['seeds']
    tasks = {}
    for task in [t['id'] for t in cfg['tasks']]:
        per = {g: [table[(task, g, s)]['last5'] for s in seeds] for g in ORDER}
        per['SEL'] = [table[(task, choice[(task, s)]['chosen'], s)]['last5'] for s in seeds]
        per['SEL_short'] = [table[(task, choice[(task, s)]['chosen_short'], s)]['last5'] for s in seeds]
        per['ORACLE_seed'] = [max(table[(task, g, s)]['last5'] for g in ORDER) for s in seeds]
        groups = {}
        for g, v in per.items():
            mean, sd, se = stats(v)
            groups[g] = dict(per_seed=dict(zip(map(str, seeds), v)), mean=mean, sd=sd, se=se, worst=min(v))
        for g in ORDER:
            groups[g]['length_mean'] = sum(table[(task, g, s)]['length_mean'] for s in seeds)/len(seeds)
            groups[g]['fraction_under_200'] = sum(table[(task, g, s)]['fraction_under_200'] for s in seeds)/len(seeds)
        strong = max(cfg['controls'], key=lambda g: groups[g]['mean'])
        cands = {}
        for c in cfg['candidates']:
            vs = paired(per[c], per[strong])
            cands[c] = dict(minus_strong=vs, screen_effective=vs['mean'] > 0 and vs['mean'] > vs['se'],
                            two_se_descriptive=vs['mean'] > 0 and vs['mean'] > 2*vs['se'],
                            minus_BC=paired(per[c], per['W0']), minus_BCMIX=paired(per[c], per['W100']))
        sel_rows = []
        for s in seeds:
            res = choice[(task, s)]
            true = [table[(task, g, s)]['last5'] for g in ORDER]
            score = [res['scores'][g]['score'] for g in ORDER]
            best = ORDER[max(range(4), key=lambda i: (true[i], -i))]
            sel_rows.append(dict(seed=s, chosen=res['chosen'], chosen_short=res['chosen_short'], best_true=best,
                                 spearman=spearman(score, true), spearman_short=spearman([res['scores'][g]['score_short'] for g in ORDER], true),
                                 model_scores={g: res['scores'][g]['score'] for g in ORDER},
                                 model_lengths={g: res['scores'][g]['length_mean'] for g in ORDER},
                                 ensemble_spread={g: res['scores'][g]['ensemble_spread'] for g in ORDER},
                                 true_last5=dict(zip(ORDER, true)), regret=max(true)-table[(task, res['chosen'], s)]['last5'],
                                 validation_error_FT=res['validation_error_FT_mean'],
                                 validation_error_source_only=res['validation_error_source_only_mean']))
        sel_vs_strong = cands['SEL']['minus_strong']
        tasks[task] = dict(groups=groups, strong_control=strong, candidates=cands, selection=sel_rows,
                           spearman_mean=sum(r['spearman'] for r in sel_rows)/len(sel_rows),
                           robust_SEL=sel_vs_strong['mean'] >= -sel_vs_strong['se'],
                           BCMIX_minus_BC=paired(per['W100'], per['W0']))
    verdicts = {}
    for c in cfg['candidates']:
        eff = [t for t in tasks if tasks[t]['candidates'][c]['screen_effective']]
        nonpos = [t for t in tasks if tasks[t]['candidates'][c]['minus_strong']['mean'] <= 0]
        verdict = 'supported' if len(eff) >= 2 else ('rejected' if len(nonpos) >= 2 else 'uncertain')
        verdicts[c] = dict(verdict=verdict, screen_effective_tasks=eff, nonpositive_tasks=nonpos)
    secondary = dict(SEL_robust_all_tasks=all(tasks[t]['robust_SEL'] for t in tasks),
                     SEL_minus_BCMIX_walker_medium=tasks.get('walker2d-kinematic-thighjnt-medium', {}).get('candidates', {}).get('SEL', {}).get('minus_BCMIX'),
                     spearman_mean_by_task={t: tasks[t]['spearman_mean'] for t in tasks})
    return runs, tasks, verdicts, secondary


def report(cfg, tasks, verdicts, secondary, elapsed):
    zh = {'supported': '支持', 'rejected': '否定', 'uncertain': '不确定'}
    lines = ['# %s：任务层面选择源数据权重（模型选权重 SEL；固定权重 W25、W50）筛查' % cfg['job'], '',
             '种子 %s（新种子）；三个任务、四个源权重 w ∈ {0, 0.25, 0.5, 1} 都在同一张卡、同一批种子下训练，w=0 即 BC，w=1 即 BC-混合。'
             '主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。SEL 不训练新策略：每个（任务、种子）用目标动力学模型'
             '（源数据预训练 + 目标数据微调的 5 个模型）从重置分布的起点模拟四个策略，选模拟回报最高的那个 w，报告它在真实环境里的 Last5。' % cfg['seeds'], '',
             '## 预设规则（提交前写进 research/PLAN.md）', '']
    lines += ['- %s：%s' % (k, v) for k, v in cfg['rule'].items()]
    lines += ['', '## 判定', '']
    for c, v in verdicts.items():
        lines.append('- %s：**%s**（筛查有效的任务：%s；均值差 ≤ 0 的任务：%s）' % (
            c, zh[v['verdict']], '、'.join(SHORT[t] for t in v['screen_effective_tasks']) or '无',
            '、'.join(SHORT[t] for t in v['nonpositive_tasks']) or '无'))
    w = secondary['SEL_minus_BCMIX_walker_medium']
    lines.append('- 次要（登记，不计入循环判定）：SEL 在三个任务上都不低于强对照 1 个配对 SE 以上：%s；walker medium 上 SEL − BC-混合 %s。' % (
        '是' if secondary['SEL_robust_all_tasks'] else '否',
        '%.2f（SE %.2f，%d/5 正）' % (w['mean'], w['se'], w['positive']) if w else '无'))
    for task, t in tasks.items():
        lines += ['', '## %s（强对照：%s，%.2f）' % (task, NAMES[t['strong_control']], t['groups'][t['strong_control']]['mean']), '',
                  '|组|Last5均值|SD|最差种子|各种子|', '|---|---|---|---|---|']
        for g in ORDER+('SEL',):
            r = t['groups'][g]
            lines.append('|%s|%.2f|%.2f|%.2f|%s|' % (NAMES[g], r['mean'], r['sd'], r['worst'], ' / '.join('%.1f' % x for x in r['per_seed'].values())))
        for g, name in (('SEL_short', 'SEL（描述：200 步模拟回报选）'), ('ORACLE_seed', '每个种子事后最优 w（描述，不可实现）')):
            r = t['groups'][g]
            lines.append('|%s|%.2f|%.2f|%.2f|%s|' % (name, r['mean'], r['sd'], r['worst'], ' / '.join('%.1f' % x for x in r['per_seed'].values())))
        lines += ['', '|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC|相对BC-混合|', '|---|---|---|---|---|---|---|---|']
        for c, r in t['candidates'].items():
            m = r['minus_strong']
            lines.append('|%s|%.2f|%.2f|%d|%s|%s|%.2f（SE %.2f）|%.2f（SE %.2f）|' % (
                c, m['mean'], m['se'], m['positive'], '是' if r['screen_effective'] else '否', '是' if r['two_se_descriptive'] else '否',
                r['minus_BC']['mean'], r['minus_BC']['se'], r['minus_BCMIX']['mean'], r['minus_BCMIX']['se']))
        lines += ['', '模型选择（描述）：', '', '|种子|选中|真实最优|Spearman|模型分数 W0/W25/W50/W100|真实 Last5 W0/W25/W50/W100|模型回合长度|微调后/只用源模型的留出误差|',
                  '|---|---|---|---|---|---|---|---|']
        for r in t['selection']:
            lines.append('|%d|%s|%s|%.2f|%s|%s|%s|%.3f / %.3f|' % (
                r['seed'], r['chosen'], r['best_true'], r['spearman'], ' / '.join('%.1f' % r['model_scores'][g] for g in ORDER),
                ' / '.join('%.1f' % r['true_last5'][g] for g in ORDER), ' / '.join('%.0f' % r['model_lengths'][g] for g in ORDER),
                r['validation_error_FT'], r['validation_error_source_only']))
        b = t['BCMIX_minus_BC']
        lines += ['', '描述性：Spearman 平均 %.2f；BC-混合 − BC %.2f（SE %.2f，%d/5 正）。' % (t['spearman_mean'], b['mean'], b['se'], b['positive'])]
    lines += ['', '## 运行情况', '', '总耗时 %.1f 小时。冒烟运行的回报不读取、不报告。所有种子都保留。' % (elapsed/3600)]
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    started = time.time()
    OUT.mkdir(exist_ok=True)
    cfg = read(HERE/'config.json')
    gpus = ['cpu'] if LOCAL_TEST else visible_gpus()
    log('GPUs', gpus)
    vendor, algo, data, sources, source_sha = prepare(cfg)
    base = HERE/('runs_'+time.strftime('%Y%m%d_%H%M%S'))
    write(OUT/'STATUS.json', dict(time=utc(), phase='smoke', runs_dir=str(base), gpus=gpus))
    for smoke in (True, False):
        tag = 'smoke' if smoke else 'main'
        stage1 = plan_stage1(cfg, vendor, algo, data, sources, source_sha, smoke)
        if not smoke:
            write(OUT/'PLAN.json', [dict(id=j['id'], kind=j['kind']) for j in stage1])
            write(OUT/'STATUS.json', dict(time=utc(), phase='main_stage1', runs=len(stage1), runs_dir=str(base)))
        done, failed = run_queue(cfg, stage1, base/tag, gpus, tag+'_stage1')
        if failed:
            write(OUT/'STATUS.json', dict(time=utc(), phase=tag+'_stage1_failed', failed=failed, completed=len(done)))
            log(tag, 'stage 1 has failed runs; stop', failed)
            return 1
        select = plan_select(cfg, stage1, done, vendor, algo, data, sources, source_sha, smoke)
        write(OUT/'STATUS.json', dict(time=utc(), phase=tag+'_select', runs=len(select), runs_dir=str(base)))
        sel_done, failed = run_queue(cfg, select, base/(tag+'_select'), gpus, tag+'_select')
        if failed:
            write(OUT/'STATUS.json', dict(time=utc(), phase=tag+'_select_failed', failed=failed, completed=len(sel_done)))
            log(tag, 'selection has failed runs; stop', failed)
            return 1
        if smoke:
            for job in stage1:
                c = read(done[job['id']]/'COMPLETED.json')
                assert c.get('completed', True) and (job['kind'] != 'policy' or (c['finite'] and c['evaluations'] == 1)), job['id']
            log('smoke passed', len(done)+len(sel_done))
    policy_jobs = [j for j in stage1 if j['kind'] == 'policy']
    runs, tasks, verdicts, secondary = summarize(cfg, policy_jobs, done, select, sel_done)
    with (OUT/'RUNS.csv').open('w', newline='', encoding='utf-8') as f:
        keys = ['task', 'group', 'seed', 'last5', 'final', 'length_mean', 'fraction_under_200', 'wall_seconds']
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in runs:
            w.writerow(r)
    write(OUT/'CURVES.json', {j['id']: collect(done[j['id']], j)['curve'] for j in policy_jobs})
    write(OUT/'SELECTION.json', {j['id']: read(sel_done[j['id']]/'COMPLETED.json')['result'] for j in select})
    write(OUT/'SOURCE_MODELS.json', {j['id']: read(done[j['id']]/'COMPLETED.json')['result'] for j in stage1 if j['kind'] == 'source'})
    write(OUT/'summary.json', dict(job=cfg['job'], seeds=cfg['seeds'], rule=cfg['rule'], tasks=tasks, verdicts=verdicts,
                                   secondary=secondary, elapsed_hours=(time.time()-started)/3600))
    report(cfg, tasks, verdicts, secondary, time.time()-started)
    write(OUT/'STATUS.json', dict(time=utc(), phase='completed', runs=len(stage1)+len(select)))
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
