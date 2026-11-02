# -*- coding: utf-8 -*-
"""Finalize stage0 with cache provenance, a Chinese report and read-only hashes."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/v9_hopper_label_headroom_20261030'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open('\\\\?\\'+str(Path(p).resolve()),'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write(name,value):
    with open(OUT/name,'x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    extra=[ROOT/'oracle_diag/v2/stage_c.py',ROOT/'oracle_diag/v2/support.py',ROOT/'oracle_diag/verify_replay_v2.py',ROOT/'oracle_diag/common.py']
    extra_hash={str(p):sha(p) for p in extra}
    # Read NPY headers inside all project NPZ archives, without decompressing tensors.
    archives=[];full_candidates=[]
    for current,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','node_modules','site-packages','.venv','Lib'} and str((Path(current)/d).resolve())!=str(OUT)]
        for name in files:
            if not name.endswith('.npz'):continue
            path=Path(current)/name;headers={}
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    if not info.filename.endswith('.npy'):continue
                    with z.open(info) as f:
                        version=np.lib.format.read_magic(f)
                        shape,_,dtype=np.lib.format._read_array_header(f,version)
                    headers[info.filename[:-4]]=dict(shape=list(shape),dtype=str(dtype))
            archives.append(dict(path=str(path),headers=headers))
            if any(h['shape']==[999998,11] for k,h in headers.items() if k in ('oracle_next','next_state','next_observations')):
                full_candidates.append(str(path))
    oldsample=ROOT/'oracle_diag/results/v2_20261006/stage_c/hopper_morph_source_oracle_fixed.npz'
    target=ROOT/'reproductions/OTDF/datasets/hopper_morph_medium.hdf5'
    with np.load(str(oldsample),allow_pickle=False) as z,h5py.File(target,'r') as f:
        exact=bool(np.array_equal(z['source_state'],f['observations'][:]) and np.array_equal(z['source_action'],f['actions'][:]))
    assert exact and not full_candidates
    inv=read(OUT/'ORACLE_CACHE_INVENTORY.json')
    assert not inv['full_hopper_morph_cache_found']
    write('CACHE_PROVENANCE_CLARIFICATION.json',dict(
        desired_direction='normal Hopper D4RL source999998 canonical rows -> Hopper-morph target dynamics',
        complete_accepted_desired_cache_found=False,all_project_npz_headers_inspected=len(archives),full_NPZ_candidates=full_candidates,
        npz_headers=archives,old5000_input_matches_morph_target_dataset=exact,
        old5000_direction='Hopper-morph dataset5000 states/actions -> normal Hopper dynamics; opposite of desired normal->morph direction',
        old5000_path=str(oldsample),old5000_sha256=sha(oldsample),provenance_code_sha256=extra_hash,
        source5000_factual_replay='normal Hopper source sample; source-dynamics factual replay only, not morph target labels',
        canonical_source_index_file='v3 stage_a1 hopper_source_raw_indices.npy is row mapping only, not oracle labels',
        new_oracle_rows=0,environment_steps=0))

    data=read(OUT/'TARGET_TRAJECTORIES.json');hist=read(OUT/'T_HISTORY.json')
    curve=read(OUT/'HOPPER_T_CURVE_DIAGNOSTICS.json')['seeds'];task=read(OUT/'TASK_DEFINITION_CHECK.json')
    original_manifest=read(OUT/'INPUT_MANIFEST.json')['sha256']
    settings={}
    for label,h in hist.items():
        paths=[Path(r['out'])/'config.json' for r in h['runs']]
        configs=[read(p) for p in paths]
        for c in configs:
            assert c['job']['temp']==10 and c['job']['steps']==500000
            assert c['batch_target']==64 and c['batch_source']==0
            assert not c['V_source'] and not c['Q_source'] and not c['actor_source']
            assert c['actor_advantage']=='pre-V-step target advantage' and c['target_Q_update']=='every gradient update'
            assert c['learner']['max_step']==500000 and c['learner']['temp']==10
            assert c['learner']['lam']==.7 and c['learner']['gamma']==.99 and c['learner']['tau']==.005
            assert c['learner']['critic_lr']==c['learner']['actor_lr']==.0003
            assert c['learner_observations']=='raw' and c['reward_scale']==1.
        settings[label]=dict(passed=True,config_paths=[str(p) for p in paths],temperature=10,updates=500000,
            target_batch=64,source_batch=0,source_in_V_Q_actor=False,actor_advantage='pre-V',cosine_max_step=500000,
            inactive_source_coefficient=[c['source_coefficient'] for c in configs])
    write('T_SETTINGS_CHECK.json',dict(groups=settings,new_paired_groups_run=False,
        inactive_source_coefficient_difference='R11/13 store source coefficient1 but source batch0 and all source channels false; R14 T stores0. Both T have zero source loss. No claim that nonmatching active source losses are equivalent.'))
    f=lambda x:'%.5f'%x
    lines=['# V9阶段0：CPU只读诊断报告','',
        '**阶段0已完成并停止。0训练、0优化更新、0模拟器步、0新oracle标签；阶段1未启动。**','',
        '本阶段由用户在2026-10-30明确授权。仅读取既有数据、XML、代码、原始TB/episode日志和缓存文件；所有新代码及报告位于新目录。',
        '五项结论：Hopper形态定义匹配OTDF附录；两个目标集均为7个完整边界段加1个EOF截断段；Hopper205持续偏低、206波动且略有回落；R12的T完全复用R11；所需Hopper全量oracle缓存尚不存在。','',
        '## 1. 任务定义与归一化核对','',
        '论文附录C.4（PDF第23页）的Hopper-morph头部几何与本地XML逐项匹配：','',
        '|字段|论文|本地|核对|','|---|---|---|---|',
        '|geom|torso_geom|torso_geom|一致|','|size|0.125|0.125|一致|',
        '|fromto|0 0 1.45 0 0 1.05|0 0 1.45 0 0 1.05|一致|',
        '|type / friction|capsule / 0.9|capsule / 0.9|一致|','',
        '本地timestep=0.002，XML SHA256与R14冻结值一致。本次未发现会使R14不能与论文直接比较的头部尺寸偏差；这一核对并不把不同学习器、超参数和评估协议宣称为完整论文复现。',
        '归一化统一由本地OTDF/info.py读取：(原始episode回报−min)/(max−min)×100，不截断分数。Walker min=10.079455055289959、max=4398.43；Hopper min=−26.3360015397715、max=3152.75。','',
        '来源：[OTDF正式论文](%s)，附录C.4与表4；本地原文与渲染证据见sources/OTDF_page_22.png、OTDF_page_23.png。'%task['official_paper_url'],
        '本地PDF SHA256：`%s`。'%task['paper_sha256'],'',
        '## 2. 目标数据轨迹与回报','',
        '**两个5000行HDF5均无timeouts字段。** 因而没有把缺失字段假装成观测到的timeout。先保存按原terminal标记分段的原始视图；再单列由explicit_next[i]与observations[i+1]不连续恢复的reset边界。',
        'Walker的额外不连续在0-based行2257、3724、4724，恢复后各对应1000步段；阈值0、1e−6、1e−5、1e−4给出完全相同边界。Hopper的所有不连续均与terminal重合。原数据与terminal语义从未改写。',
        '下表为恢复后的逐段结果。完整边界段指结束于terminal或推断的1000步reset；文件开头视为数据起点，未另存reset元数据，不能额外证明其之前没有缺失前缀。末尾段明确不完整，其partial return不能当作完整episode回报。','',
        '|任务|段|原始行范围（含端点）|长度|原始回报|归一化回报|结束证据|',
        '|---|---:|---|---:|---:|---:|---|']
    reason={'terminal':'terminal','inferred_reset_at_1000':'推断1000步reset（非原始timeout列）','EOF_incomplete':'EOF不完整'}
    for name,d in data.items():
        for r in d['reconstructed_segments']:
            lines.append('|%s|%d|%d–%d|%d|%s|%s|%s|'%(name,r['segment'],r['start_row'],r['end_row'],r['length'],f(r['raw_return']),f(r['normalized_return']),reason[r['end_reason']]))
    lines+=['','按条数取ceil(n×比例)、至少1条；按回报降序，相同回报保持原段顺序。包括尾段与剔除尾段两种汇总均保留。以下为归一化回报：','',
        '|任务/统计集合|n|均值|中位数|最小|最大|样本SD|Q1|前25%均值（条数）|前50%均值（条数）|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,d in data.items():
        for view,label in [('completed_boundary_segments','完整边界段'),('including_EOF_fragment','包括EOF片段')]:
            a=d['summaries'][view]['normalized']
            lines.append('|%s / %s|%d|%s|%s|%s|%s|%s|%s|%s（%d）|%s（%d）|'%(name,label,a['n'],f(a['mean']),f(a['median']),f(a['minimum']),f(a['maximum']),f(a['sample_SD']),f(a['Q1']),f(a['top25_mean']),a['top25_count'],f(a['top50_mean']),a['top50_count']))
    lines+=['','完整边界段的原始回报汇总：','',
        '|任务|均值|中位数|最小|最大|前25%均值|前50%均值|','|---|---:|---:|---:|---:|---:|---:|']
    for name,d in data.items():
        a=d['summaries']['completed_boundary_segments']['raw']
        lines.append('|%s|%s|%s|%s|%s|%s|%s|'%(name,*[f(a[k]) for k in ('mean','median','minimum','maximum','top25_mean','top50_mean')]))
    lines+=['','Hopper完整7段平均原始回报1980.33477，对应63.12100；论文表4为1980.33，换算63.12085，吻合。含436步末尾片段的8段平均仅59.97323，不能与论文完整轨迹均值混用。',
        'Walker论文表4平均2013.11，本地完整段2013.65854；差约0.55原始回报（约0.0125归一化分），不将其称为逐值完全相同。这里报告实际本地文件统计，不修改数据或归一化常数。',
        '完整轨迹均值只是既有行为数据的描述，不能证明BC可达到该分数或证明critic已是瓶颈。阶段3未执行，不代替其判定。',
        '**阶段3尚未确定使用n=7完整段还是包括末尾片段的n=8。** 本阶段只提供两种账本，不擅自修改后续训练集规则；两种n均小于20，若按现有定义均不触发%BC-10。','',
        '## 3. Hopper T训练曲线：205、206为何偏低','',
        '已从R14各seed的50个评估点、每点10回合原始回报重算，并逐点与TensorBoard交叉核对。没有平滑，没有重跑，没有据峰值挑检查点。峰值与滚动均值仅作事后曲线描述。','',
        '![Hopper全部T曲线](figures/hopper_T_all_seeds_v3.png)','',
        '|seed|10–100k均值|110–200k|210–300k|310–400k|410–500k|Last5|Final|单点最大值/步数|最高连续5点均值/窗口|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for seed in ('205','206'):
        a=curve[seed]
        lines.append('|%s|%s|%s|%s|%s|%s|%s|%s|%s / %dk|%s / %d–%dk|'%(seed,*[f(v) for v in a['block_means']],f(a['Last5']),f(a['Final']),f(a['maximum_single_point']),a['maximum_single_step']//1000,f(a['maximum_5point_mean']),a['maximum_5point_window'][0]//1000,a['maximum_5point_window'][1]//1000))
    lines+=['','**205：主要是一直相对偏低并缓慢改善。** 420k单点37.31后回落，但最高连续5点评估均值仅24.19，最后19.24；不能把一个短暂峰值写成先稳定学到高水平再崩溃。',
        '**206：早期提升后在较低平台波动，存在一定回落。** 180–220k最高连续5点30.89，Last5为25.30，回落5.58；它不是严格单调、也非完全没有回落，但曲线不支持从稳定高分突然全面崩溃的叙述。',
        '上述是描述性读图，不新增正式“崩溃”阈值或显著性检验；10seed全部保留，不能删除205、206来改善T或source组结果。','',
        '## 4. 已有T结果、复用关系与配置','',
        '主指标为460/470/480/490/500k五点评估均值；SD为训练seed样本SD(ddof1)，Q1线性25%分位。Final仅辅助。','',
        '|组别|seeds|Last5均值|样本SD|最差分数/seed|Q1|Final均值|Final SD|Final最差/seed|Final Q1|',
        '|---|---|---:|---:|---|---:|---:|---:|---|---:|']
    for name,h in hist.items():
        a=h['Last5'];b=h['Final']
        lines.append('|%s|%s|%s|%s|%s / %s|%s|%s|%s|%s / %s|%s|'%(name,','.join(map(str,h['seeds'])),f(a['mean']),f(a['sample_SD']),f(a['minimum']),','.join(map(str,a['worst_seeds'])),f(a['Q1']),f(b['mean']),f(b['sample_SD']),f(b['minimum']),','.join(map(str,b['worst_seeds'])),f(b['Q1'])))
    lines+=['','**R12没有新T实验：它复用R11的同3个文件（seeds101–103），不能算另一批独立证据。R13为新seed201–205，R14 Walker又复用同一R13 T5。Hopper T为R14独立201–210。** R11与R13不同seed，不作同seed配对差；本阶段也未新增训练条件，配对均值/SE/正号不适用。','',
        '|组别|原始出处|','|---|---|',
        '|R11 Walker T|oracle_diag/results/coverage_oracle_20261016_r11/runs/tune_T_t10_h500000_s101…103；选择凭据coverage_oracle_20261016_r11_resume2/T_selected.json|',
        '|R12 Walker T|coverage_oracle_20261018_r12/T_selected.json，逐项指向上述R11路径|',
        '|R13 Walker T|coverage_oracle_20261021_r13/runs/confirm_T_t10_h500000_s201…205；finish1复用收据确认|',
        '|R14 Hopper T|coverage_raw_20261026_r14/runs/hopp_w0_s201…210|','',
        '逐seed全部数值、绝对路径、audit与评估hash见T_history_per_seed.csv和T_HISTORY.json。','',
        '|配置项|R11/R12/R13 Walker T|R14 Hopper T|',
        '|---|---|---|','|温度/更新/actor cosine周期|10 / 500k / 500k|10 / 500k / 500k|',
        '|批大小/通道|64 target，0 source；V/Q/actor均无source|相同|',
        '|actor advantage / target Q|pre-V target advantage / 每梯度更新|相同|',
        '|γ / expectile / τ|.99 / .7 / .005|相同|','|actor/critic学习率|均3e−4|相同|',
        '|观测/奖励|原始观测，无新归一化，reward scale1|相同|','|任务维度|17/6|11/3|','',
        '旧T配置中source_coefficient虽存为1，但source batch=0且三个source通道均关闭，没有source损失；R14 T存为0。该无效占位值差异已单列，不把旧T错当作w=1混合组。阶段0没有创建需与R14配对的新训练组，也没有声称跨任务配对。','',
        '## 5. Hopper oracle缓存状态与方向','',
        '在整个项目（排除.git、Python环境/依赖目录与本次新输出）搜索标准NPY缓存，并检查全部%d个NPZ的数组头。未找到999998×11行的完整Hopper-morph源侧oracle缓存及通过验收的完成凭据。'%len(archives),
        '本次所需方向为：**正常Hopper源数据999998 canonical行 → Hopper-morph目标动力学标签**。','',
        '|现有文件/凭据|实际内容|能否替代所需全量缓存|','|---|---|---|',
        '|v3/source_oracle1|999322×17，Walker-morph全量缓存|不能，任务不同|',
        '|v3/stage_a2/hopper_source_acceptance.json|正常Hopper source的5000点事实回放验收|不能，未生成morph反事实标签|',
        '|v3/stage_a1/hopper_source_raw_indices.npy|999998行canonical原始索引映射|不能，仅索引|',
        '|v2/stage_c/hopper_morph_source_oracle_fixed.npz|5000个Hopper-morph目标数据状态/动作，反事实回放到正常Hopper|不能，输入集合与动力学方向均不同|',
        '|v3/stage_a1/hopper_morph_target_replay.npz|5000行morph目标数据的事实回放|不能，是target数据验收|','',
        '方向核验同时依据source_state/source_action与morph目标HDF5逐位相同，以及v2/stage_c.py、v2/support.py、verify_replay_v2.py与common.make_target_env的代码路径；不以文件名中的source/oracle字样判断。本阶段未创建、修复或补齐任何缓存。','',
        '## 6. 复核、异常、命令与下一阶段','',
        '主诊断脚本耗时%.2f秒；文献联网核查、PDF只读渲染、绘图和人工复核另计。新训练/优化更新/模拟器步/新oracle行均为0。'%read(OUT/'READ_ONLY_VERIFICATION.json')['elapsed_seconds'],
        '异常与偏离：目标文件缺少timeouts，故明确区分观测标记与连续性推断；原指令从桌面移至Desktop/Claude命令后按同名文件定位并保存hash；首次仅因旧指令路径不可用而在任何统计输出前退出，没有重训。',
        '只读PDF提取曾遇控制台编码问题，改用UTF-8；绘图前两版存在标题/坐标标签覆盖，已保留版本并在第三版显式关闭自动布局后通过程序与彩色/灰度目视检查。旧实验产物未覆盖。',
        '本阶段没有A/B标签效应判定、没有BC上限H判定、没有近/远选择判定；这些均须后续授权和数据。未改变V9任何决策规则。','',
        '代码目录：`%s`。输出目录：`%s`。'%(CODE,OUT),
        '配置：config.json、PLOT_CONFIG_v3.json；数据/版本保护：INPUT_MANIFEST.json、OLD_FILE_PROTECTION_BEFORE.json；最终凭据：STAGE0_COMPLETED.json。','',
        '实际命令（工作目录为项目根目录；所有Python均-B，CUDA_VISIBLE_DEVICES=-1、PYTHONDONTWRITEBYTECODE=1）：','',
        '```powershell',
        "& 'D:/ANACONDA/envs/offdynamics/python.exe' -B 'oracle_diag/v9_hopper_label_headroom/stage0_diagnose.py'",
        "& 'C:/Users/TXin/.codex/environments/scientific-figures/.venv/Scripts/python.exe' -B 'oracle_diag/v9_hopper_label_headroom/stage0_plot.py'",
        "& 'C:/Users/TXin/.codex/environments/scientific-figures/.venv/Scripts/python.exe' -B 'oracle_diag/v9_hopper_label_headroom/stage0_plot.py' --vector-after-qa",
        "& 'D:/ANACONDA/envs/offdynamics/python.exe' -B 'oracle_diag/v9_hopper_label_headroom/stage0_finalize.py'",
        '```','',
        'PDF只读渲染使用bundled Poppler pdftoppm（第22、23页）；图的程序合规检查使用scipilot-figure-skill/scripts/check_figure.py --strict，最终PNG与SVG均PASS。','',
        '**按用户本轮授权，阶段0到此停止，等待阶段1的明确确认。** 阶段1若获批，仅生成Hopper正常源数据在morph动力学下的全量oracle标签，并独立回放至少10000条验收，不包括任何训练。',
        '阶段1暂预留约0.5–2小时CPU作为规划量，并非Hopper实测承诺：历史Walker全量生成记录675.05秒（约11.25分钟），Hopper隐藏状态恢复、失败诊断及独立验收耗时尚未知；验收不通过则停止，不能进入阶段2。阶段2/3/4均未启动。','']
    report=OUT/'STAGE0_REPORT_ZH.md'
    with open(report,'x',encoding='utf-8') as f:f.write('\n'.join(lines))
    # Final, content-based preservation check after all reading and plotting.
    protected=read(OUT/'OLD_FILE_PROTECTION_BEFORE.json')['sha256']
    changes=[p for p,h in {**protected,**original_manifest,**extra_hash}.items() if sha(p)!=h]
    assert not changes,changes
    plotcfg=read(OUT/'PLOT_CONFIG_v3.json');assert not plotcfg['layout_issues']
    assert plotcfg['code_sha256']==sha(CODE/'stage0_plot.py')
    write('FIGURE_VISUAL_QA.json',dict(passed=True,final_version=3,color_and_grayscale_reviewed=True,no_clipping=True,
        no_title_or_axis_overlap=True,all10seeds_visible=True,shared_axes=True,unsmoothed=True,program_layout_issues=[],
        old_attempts_preserved=True,PNG_and_SVG_compliance='PASS'))
    outputs={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and 'matplotlib_cache' not in str(p)}
    write('STAGE0_COMPLETED.json',dict(passed=True,status='AWAITING_STAGE1_CONFIRMATION',CPU_only=True,
        training_runs=0,optimization_updates=0,simulator_steps=0,new_oracle_rows=0,next_stage_started=False,
        old_protected_count=len(protected),read_input_count=len(original_manifest),changed_old_or_input_files=[],
        report=str(report),report_sha256=sha(report),code_sha256={str(p):sha(p) for p in CODE.glob('*.py')},
        output_sha256=outputs,additional_input_sha256=extra_hash,
        normalization_reference='OTDF/info.py',timeout_metadata_missing_disclosed=True,oracle_cache_direction_disambiguated=True,
        final_figure_version=3,verified_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))
    print(json.dumps(dict(passed=True,stage=0,old_protected=len(protected),new_training=0,report=str(report)),ensure_ascii=False))

if __name__=='__main__':main()
