"""Build the adaptive-source job directories (j013-j016) from methods/adaptive_source/src.

Every job gets a byte-identical copy of all shared files (src/), its own config.json and job.json, and a
FROZEN.json listing the SHA-256 of every file under code/. Jobs that need the official hopper-morph target data
(j013, j015a) also get otdf_official/datasets/hopper_morph_medium.hdf5 (official repository, commit 41e2e62).

usage:
  python build_jobs.py --hopper-morph-hdf5 <path> [--j014-layout single|split] [--only j013,j015a,...] [--out <repo root>]
The default layout is one j014 job per (seed, task) with all seven groups; 'split' makes two jobs per (seed, task)
that both contain rho80 (A = rho80, rho0, rho20T, BC; B = rho80, rho100T, rho80T, rho80R).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
SRC = HERE/'src'
REPO = HERE.parent.parent
HOPPER_MORPH_SHA256 = '135d69d7ae3d20d179219b0ae4f489d77e29b651375a7e66f4a3f1b5d85cd7bb'
SEEDS = [3129, 3130, 3131, 3132, 3133]
SMOKE_SEED = 2999
PREFLIGHT = ['j013_adapt_preflight', 'j013b_adapt_preflight', 'j013c_adapt_preflight']
DYN = ['j015a_dyn_models', 'j015a2_dyn_models', 'j015a3_dyn_models']
SELECT = ['j015b_offline_select', 'j015b2_offline_select']
COMMON = dict(max_per_gpu=6, mem_limit=0.75, external_retry_max=3, launch_gap_seconds=20)
LAYER = dict(layer_weights=dict(target=1.0, far=0.5, near=0.5), near_batch=64)
GROUPS = {
    'rho0': dict(name='rho0', learner='OTDF', proportion=0.0),
    'rho80': dict(name='rho80', learner='OTDF', proportion=0.8),
    'rho20T': dict(name='rho20T', learner='OTDF', proportion=0.2),
    'rho100T': dict(name='rho100T', learner='OTDF', proportion=1.0),
    'rho80T': dict(name='rho80T', learner='OTDF', proportion=0.8),
    'rho80R': dict(name='rho80R', learner='OTDF', proportion=0.8),
    'BC': dict(name='BC', learner='BC'),
    'IQLCAL': dict(name='IQLCAL', learner='OTDF', proportion=0.0, noreg=True),
}
ORDER7 = ['rho0', 'rho80', 'rho20T', 'rho100T', 'rho80T', 'rho80R', 'BC']
SPLIT = {'A': ['rho80', 'rho0', 'rho20T', 'BC'], 'B': ['rho80', 'rho100T', 'rho80T', 'rho80R']}
CANDIDATES = ['rho0', 'rho20T', 'rho80', 'rho100T', 'BC']
SUPPORT_DEFINITION = ('Distances are Euclidean in the space standardised by the per-dimension mean and standard deviation (floor 1e-3) '
                      'of the target training arrays (official call_tar_dataset rows). tau = 0.9 quantile of the target (s, a) '
                      'leave-one-trajectory-out nearest-neighbour distances (all points of the own trajectory excluded); tau_state: '
                      'the same with states only. D_ret(rho) = first floor(rho * N) source rows sorted by the stored OT value, '
                      'largest first (stable); D_near(rho) = rows of D_ret(rho) with (s, a) distance <= tau; rho80R = |D_near(0.8)| '
                      'rows drawn without replacement from D_ret(0.8) with a fixed per-task seed.')


def sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def shared_files():
    out = []
    for p in sorted(SRC.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc'):
            out.append(p.relative_to(SRC))
    return out


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def build(dest, job, config, extra=None):
    code = dest/'code'
    if dest.exists():
        shutil.rmtree(str(dest))
    code.mkdir(parents=True)
    files = shared_files()
    for rel in files:
        (code/rel).parent.mkdir(parents=True, exist_ok=True)
        data = (SRC/rel).read_bytes()
        assert b'\r\n' not in data or rel.suffix in ('.hdf5', '.xml'), 'CRLF in shared text file %s' % rel
        (code/rel).write_bytes(data)
    (code/'run.sh').chmod(0o755)
    for rel, src in (extra or {}).items():
        (code/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(src), str(code/rel))
    write_json(code/'config.json', config)
    shared = {str(r).replace('\\', '/'): sha(SRC/r) for r in files}
    allf = {}
    for p in sorted(code.rglob('*')):
        if p.is_file() and p.name != 'FROZEN.json':
            allf[str(p.relative_to(code)).replace('\\', '/')] = sha(p)
    digest = hashlib.sha256(json.dumps(shared, sort_keys=True).encode()).hexdigest()
    write_json(code/'FROZEN.json', dict(job=config['job'], shared_files_digest=digest, files=allf))
    write_json(dest/'job.json', job)
    return digest


def job_json(name, description, hypothesis, max_hours, parent):
    return dict(job_id=name, description=description, hypothesis=hypothesis, gpus=1, max_hours=max_hours, retries=1,
                conda_env='offdynamics38', result_dirs=['outputs'], approved_by_user=False, created_by='claude', parent_job=parent)


def tasks_spec():
    return json.loads((SRC/'tasks.json').read_text(encoding='utf-8'))


def j014_layout(spec, layout):
    out = {}
    for t in spec['tasks']:
        out[t['short']] = {}
        for seed in SEEDS:
            if layout == 'single':
                out[t['short']][str(seed)] = {c: 'j014_s%d_%s' % (seed, t['short']) for c in CANDIDATES+['rho80T', 'rho80R']}
            else:
                m = {}
                for part in ('B', 'A'):                     # rho80 is read from part A (both parts contain it)
                    for g in SPLIT[part]:
                        m[g] = 'j014_s%d_%s_%s' % (seed, t['short'], part)
                out[t['short']][str(seed)] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hopper-morph-hdf5', required=True)
    ap.add_argument('--j014-layout', default='single', choices=['single', 'split'])
    ap.add_argument('--j014-max-hours', type=float, default=24.)
    ap.add_argument('--only', default='')
    ap.add_argument('--preflight-name', default=PREFLIGHT[0], choices=PREFLIGHT,
                    help='name of the preflight job (a rerun uses j013b_adapt_preflight / j013c_adapt_preflight)')
    ap.add_argument('--out', default=str(REPO))
    a = ap.parse_args()
    root = Path(a.out)
    hm = Path(a.hopper_morph_hdf5)
    assert sha(hm) == HOPPER_MORPH_SHA256, 'not the official hopper_morph_medium.hdf5'
    extra_hm = {'otdf_official/datasets/hopper_morph_medium.hdf5': hm}
    spec = tasks_spec()
    ids = [t['id'] for t in spec['tasks']]
    only = set(x for x in a.only.split(',') if x)
    digests = {}

    def want(k):
        return not only or k in only

    if want('j013'):
        pname = a.preflight_name
        cfg = dict(job=pname, kind='preflight', tasks=ids, **COMMON, **LAYER,
                   purpose='Preflight of the adaptive-source batch: hashes, server-side OT costs, support statistics, unit tests a-d, '
                           'smoke of every group on the 7 tasks, throughput of 6 OTDF processes per GPU, normalization table.',
                   ot=dict(conda_env_candidates=['otdf_ot', 'ot', 'jax'], cores_per_process=8, max_hours=6.0),
                   unit_tests=dict(steps=300, a_tasks=['hopper-kinematic-legjnt-medium', 'walker2d-kinematic-thighjnt-hard'],
                                   layer_tasks=['hopper-kinematic-legjnt-medium', 'walker2d-kinematic-thighjnt-hard']),
                   smoke=dict(seed=SMOKE_SEED, steps=2000, iqlcal_tasks=['hopper-kinematic-footjnt-medium', 'hopper-kinematic-legjnt-medium']),
                   throughput=dict(tasks=['walker2d-kinematic-thighjnt-hard', 'hopper-kinematic-legjnt-medium'], steps=20000, eval_every=10000),
                   normalization_grid=[-100, 0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000],
                   support_definition=SUPPORT_DEFINITION)
        job = job_json(pname,
                       '自适应源数据批次预检（research/PLAN.md j013 登记）：校验全部数据、XML 与 OTDF 官方文件哈希；服务器端用官方 solve_ot 计算 7 个任务的 OT 代价并核对数组哈希；'
                       '支撑统计（tau、tau_state、近邻层比例）；单元测试 a–d；7 任务×全部组别冒烟（种子 2999，不读回报）；6 进程吞吐与 j014 拆分建议；归一化对照。',
                       '预检，不检验假设、不读回报；全部检查通过（PREFLIGHT.json all_passed）才能把 staging/j014 移入 jobs/。', 8,
                       'j012d_p2base_walkg05' if pname == PREFLIGHT[0] else PREFLIGHT[PREFLIGHT.index(pname)-1])
        digests['j013'] = build(root/'jobs'/pname, job, cfg, extra_hm)
    if want('j015a'):
        cfg = dict(job='j015a_dyn_models', kind='dyn', tasks=ids, seeds=SEEDS, **COMMON,
                   model=dict(members=5, hidden=[200, 200, 200, 200],
                              source=dict(steps=200000, batch=512, lr=1e-3, validation_fraction=0.02, eval_every=5000),
                              target=dict(max_steps=30000, batch=256, lr=3e-4, eval_every=500, patience_evals=20),
                              folds_max=5, es_fraction=0.1, eps0=[0.05, 0.1, 0.2, 0.5], quantile=0.9))
        job = job_json('j015a_dyn_models',
                       '7 个任务×种子 3129–3133，每个（任务、种子）5 个概率动力学模型（(s,a)→(Δs,r) 高斯，NLL，4×200 SiLU）：源数据 20 万步预训练（同机器人共用）+ 目标数据微调；'
                       '按轨迹 min(轨迹数,5) 折交叉校准开环误差，输出 H*(ε0)（ε0 ∈ {0.05,0.1,0.2,0.5}）与误差曲线；最终模型在全部目标数据上微调并保存（含哈希）。',
                       '模型准备，不检验假设、不读回报；见 research/PLAN.md j015a 登记。', 16, 'j013_adapt_preflight')
        digests['j015a'] = build(root/'jobs'/'j015a_dyn_models', job, cfg, extra_hm)
    layout = j014_layout(spec, a.j014_layout)
    if want('j014'):
        stage = root/'staging'/'j014'
        for seed in SEEDS:
            for t in spec['tasks']:
                parts = {'': ORDER7} if a.j014_layout == 'single' else {'_A': SPLIT['A'], '_B': SPLIT['B']}
                for suffix, groups in parts.items():
                    name = 'j014_s%d_%s%s' % (seed, t['short'], suffix)
                    cfg = dict(job=name, kind='train', tasks=[t['id']], seeds=[seed], seed_origin=SEEDS[0],
                               groups=[GROUPS[g] for g in groups], otdf=dict(steps=1000000, eval_every=10000, save_last=5),
                               bc=dict(steps=200000, eval_every=5000, save_last=5), preflight_candidates=PREFLIGHT, **COMMON, **LAYER)
                    job = job_json(name, '自适应源数据候选矩阵（research/PLAN.md j014 登记）：%s × 种子 %d × {%s}；OTDF 官方超参数（每域 128 条、权重开启、CVAE 1 万步、100 万步、每 1 万步评估 10 回合，约束系数 %g），BC 20 万步；保存 Last5 五个点的策略。' % (
                        t['id'], seed, ', '.join(groups), t['reg_weight']), '候选回报矩阵，判定见 research/PLAN.md j014 登记（全部 35 个任务结束后由 analysis_tools/j014_matrix.py 统一判定）。',
                        a.j014_max_hours, 'j013_adapt_preflight')
                    digests[name] = build(stage/name, job, cfg)
        cfg = dict(job='j014c_iqlcal', kind='train', tasks=['hopper-kinematic-footjnt-medium', 'hopper-kinematic-legjnt-medium'], seeds=SEEDS,
                   seed_origin=SEEDS[0], groups=[GROUPS['IQLCAL']], otdf=dict(steps=1000000, eval_every=10000, save_last=5),
                   bc=dict(steps=200000, eval_every=5000, save_last=5), preflight_candidates=PREFLIGHT, **COMMON, **LAYER)
        job = job_json('j014c_iqlcal', '计分口径核对（research/PLAN.md j014c 登记）：OTDF proportion 0 + --noreg（官方代码中的纯 IQL），hopper-kinematic-footjnt-medium 与 hopper-kinematic-legjnt-medium × 种子 3129–3133，100 万步。',
                       '描述性，不判定。', 24, 'j013_adapt_preflight')
        digests['j014c_iqlcal'] = build(stage/'j014c_iqlcal', job, cfg)
    if want('j015b'):
        cfg = dict(job='j015b_offline_select', kind='select', tasks=ids, seeds=SEEDS, candidates=CANDIDATES, default='rho80',
                   kappas=[0, 0.5, 1, 1.5, 2, 3], fixed_horizons=[50, 100, 200, 1000], starts=dict(trajectory=50, uniform=50),
                   j014_layout=layout, dyn_candidates=DYN, preflight_candidates=PREFLIGHT, **COMMON)
        job = job_json('j015b_offline_select', '离线选择（research/PLAN.md j015b 登记）：在 j015a 的动力学模型中推演 j014 候选（ρ0、ρ20T、ρ80、ρ100T、BC）的 Last5 策略，'
                       '主截断 H*+摔倒+(s,a) 距离>tau，及两种消融与固定步长；LCB 选择（κ ∈ {0,0.5,1,1.5,2,3}，默认 ρ80）；不读任何真实回报；输出 SELECTION.json 及哈希。',
                       '选择结果的判定见 research/PLAN.md j014/j015b 登记，由 analysis_tools/j014_matrix.py 计算。', 8, 'j015a_dyn_models')
        digests['j015b'] = build(root/'staging'/'j015b_offline_select', job, cfg)
    if want('j016'):
        cfg = dict(job='j016_sim_check', kind='simcheck', tasks=ids, seeds=SEEDS, candidates=CANDIDATES, fixed_horizons=[50, 100, 200, 1000],
                   j014_layout=layout, dyn_candidates=DYN, select_candidates=SELECT, preflight_candidates=PREFLIGHT, wait_hours=10, **COMMON)
        job = job_json('j016_sim_check', '仿真器核对（仅分析，research/PLAN.md j016 登记）：从 j015b 的起点恢复仿真器状态，真实运行各候选策略 H*/50/100/200/1000 步，'
                       '比较模型估计误差、短时真实回报与 Last5 的 Spearman 相关、H*/偏离程度/误差的关系。选择过程不使用本任务结果。', '描述性分析，不判定。', 12, 'j015b_offline_select')
        digests['j016'] = build(root/'staging'/'j016_sim_check', job, cfg)
    vals = set(digests.values())
    assert len(vals) <= 1, 'shared files differ between jobs'
    print(json.dumps(dict(jobs=len(digests), shared_files_digest=sorted(vals)), indent=2))


if __name__ == '__main__':
    main()
