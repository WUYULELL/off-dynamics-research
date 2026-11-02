# -*- coding: utf-8 -*-
"""V9 stage0: local CPU reads only; no learner, simulator or optimization."""
import ast
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.dont_write_bytecode=True
import h5py
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/v9_hopper_label_headroom_20261030'
R14=ROOT/'oracle_diag/results/coverage_raw_20261026_r14'
R14END=ROOT/'oracle_diag/results/coverage_raw_20261029_r14_finish1'
INPUTS={}

def sha(p):
    h=hashlib.sha256()
    with open('\\\\?\\'+str(Path(p).resolve()),'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def track(p):
    p=Path(p).resolve();h=sha(p)
    if str(p) in INPUTS: assert INPUTS[str(p)]==h
    INPUTS[str(p)]=h
    return p

def read(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))

def write(name,obj):
    with open(OUT/name,'x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def csv_write(name,rows):
    with open(OUT/name,'x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stats(x,seeds=None):
    a=np.asarray(x,dtype=np.float64)
    r=dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),minimum=float(a.min()),maximum=float(a.max()),
        sample_SD=float(a.std(ddof=1)) if len(a)>1 else None,Q1=float(np.quantile(a,.25,interpolation='linear')),
        top25_count=max(1,math.ceil(len(a)/4)),top50_count=max(1,math.ceil(len(a)/2)))
    order=np.sort(a)[::-1]
    r.update(top25_mean=float(order[:r['top25_count']].mean()),top50_mean=float(order[:r['top50_count']].mean()))
    if seeds is not None:r['worst_seeds']=[s for s,v in zip(seeds,a) if v==a.min()]
    return r

def short(p):
    import ctypes
    buf=ctypes.create_unicode_buffer(32768)
    assert ctypes.windll.kernel32.GetShortPathNameW(str(p),buf,len(buf))
    return buf.value

def main():
    begin=time.time()
    instruction_name='给Code的指令_v9_hopper标签归因与提升空间上限.md'
    userfile=track(next(p for p in (Path.home()/'Desktop'/instruction_name,Path.home()/'Desktop/Claude命令'/instruction_name) if p.exists()))
    (OUT/'sources/V9_INSTRUCTIONS.md').write_bytes(userfile.read_bytes())
    info=track(ROOT/'reproductions/OTDF/info.py')
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(info.read_text(encoding='utf-8')).body
                 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    mn=assignments['REF_MIN_SCORE'];mx=assignments['REF_MAX_SCORE']
    def normalize(task,ret):return (ret-mn[task])/(mx[task]-mn[task])*100.
    config=dict(stage=0,authorization='2026-10-30 user explicitly requested CPU read-only stage0',
        code=str(CODE),outputs=str(OUT),CPU_only=True,CUDA_VISIBLE_DEVICES='-1',
        optimization_updates=0,training_runs=0,simulator_steps=0,new_oracle_rows=0,
        timestamps='Asia/Shanghai',normalization_info=str(info),normalization_sha256=sha(info),
        trajectory_definition='terminals OR observed timeouts; missing timeouts are never fabricated; additional reset boundaries separately inferred from explicit_next discontinuities',
        continuity_thresholds=[0.,1e-6,1e-5,1e-4],primary_inferred_reset_tolerance=1e-6,
        incomplete_tail='separate report; both including-tail and complete-boundary-only summaries',
        top_selection='descending return; ceil(n*fraction), at least1; ties original segment order',
        evaluation_steps=list(range(10000,500001,10000)),Last5=[460000,470000,480000,490000,500000],
        curve_blocks=[[10000+100000*i,100000*(i+1)] for i in range(5)],
        curve_interpretation='descriptive only; no new collapse significance gate or training selection',
        next_stage_authorized=False,source_instruction_path=str(userfile),source_instruction_sha256=sha(userfile),
        python=sys.version,numpy=np.__version__,h5py=h5py.__version__,
        code_sha256={str(p):sha(p) for p in CODE.glob('*.py')})
    write('config.json',config)
    # Preserve old results by verifying the existing protection set at stage end.
    protected=read(R14END/'protected_manifest.json')['sha256']
    # Also protect completed recovery files, which postdate that manifest.
    for directory in (R14END,ROOT/'outputs/research_idea_protocol_20261029_r14_finish1'):
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in str(p):protected[str(p.resolve())]=sha(p)
    write('OLD_FILE_PROTECTION_BEFORE.json',dict(count=len(protected),sha256=protected))

    xmlpath=track(ROOT/'reproductions/OTDF/envs/assets/hopper_morph.xml')
    xml=ET.parse(xmlpath).getroot();geom=xml.find(".//geom[@name='torso_geom']")
    fields={k:geom.attrib[k] for k in ('name','size','fromto','type','friction')}
    expected=dict(name='torso_geom',size='0.125',fromto='0 0 1.45 0 0 1.05',type='capsule',friction='0.9')
    assert fields==expected
    proto=read(R14/'protocol.json');hs=next(t for t in proto['tasks'] if t['task']=='hopper-morph')
    assert sha(xmlpath)==hs['xml_sha256'] and hs['score_min']==mn['hopper-morph'] and hs['score_max']==mx['hopper-morph']
    paper=track(ROOT/'outputs/research_idea_dir1_litcheck_20261026/papers/OTDF.pdf')
    taskcheck=dict(paper=str(paper),paper_sha256=sha(paper),official_paper_url='https://proceedings.iclr.cc/paper_files/paper/2025/file/d4ce6738e84876aa79f13c8bc8b7c5eb-Paper-Conference.pdf',
        appendix='C.4 morphology, PDF p23; Table4 PDF p22',local_xml=str(xmlpath),local_xml_sha256=sha(xmlpath),
        local_fields=fields,paper_fields=expected,all_listed_hopper_morphology_fields_match=True,
        timestep=xml.find('option').attrib['timestep'],R14_xml_sha256_matches=True,
        conclusion='No hopper head-size mismatch found; this does not establish full algorithm/protocol equivalence with the paper',
        paper_target_medium_raw=dict(minimum=1946.73,maximum=2039.80,mean=1980.33),
        paper_target_medium_mean_normalized=normalize('hopper-morph',1980.33))
    write('TASK_DEFINITION_CHECK.json',taskcheck)
    trajectories={};all_segments=[]
    for task in ('walker2d-morph','hopper-morph'):
        p=track(ROOT/('reproductions/OTDF/datasets/'+task.replace('-','_')+'_medium.hdf5'))
        with h5py.File(p,'r') as f:
            keys=list(f);obs=np.asarray(f['observations']);nxt=np.asarray(f['next_observations'])
            rew=np.asarray(f['rewards'],dtype=np.float64).reshape(-1)
            term=np.asarray(f['terminals']).reshape(-1).astype(bool)
            timeout=np.asarray(f['timeouts']).reshape(-1).astype(bool) if 'timeouts' in f else np.zeros(len(rew),bool)
        assert len(rew)==5000 and np.isfinite(rew).all()
        diff=np.max(np.abs(nxt[:-1]-obs[1:]),axis=1)
        jump=np.r_[diff>1e-6,False]
        # Stability is a diagnostic assertion, not a fitted threshold.
        assert all(np.array_equal(jump[:-1],diff>tol) for tol in (0,1e-5,1e-4))
        inferred=jump & ~(term|timeout)
        flags=term|timeout
        def segments(boundaries,view):
            rows=[];start=0
            for end in np.flatnonzero(boundaries).tolist()+([] if boundaries[-1] else [len(rew)-1]):
                complete=bool(boundaries[end]);ret=float(rew[start:end+1].sum(dtype=np.float64))
                reason='terminal' if term[end] else 'observed_timeout' if timeout[end] else 'inferred_reset_at_1000' if inferred[end] else 'EOF_incomplete'
                rows.append(dict(task=task,view=view,segment=len(rows)+1,start_row=start,end_row=end,length=end-start+1,
                    raw_return=ret,normalized_return=normalize(task,ret),end_reason=reason,complete_boundary=complete,
                    start_is_dataset_begin=start==0))
                start=end+1
            assert sum(r['length'] for r in rows)==5000
            return rows
        observed=segments(flags,'observed_flags_only')
        recovered=segments(flags|jump,'explicit_next_reset_reconstruction')
        assert all(r['length']==1000 for r in recovered if r['end_reason']=='inferred_reset_at_1000')
        complete=[r for r in recovered if r['complete_boundary']]
        summaries={}
        for label,rows in [('including_EOF_fragment',recovered),('completed_boundary_segments',complete)]:
            summaries[label]=dict(raw=stats([r['raw_return'] for r in rows]),normalized=stats([r['normalized_return'] for r in rows]),
                top25_segments=[r['segment'] for r in sorted(rows,key=lambda r:(-r['raw_return'],r['segment']))[:max(1,math.ceil(len(rows)/4))]],
                top50_segments=[r['segment'] for r in sorted(rows,key=lambda r:(-r['raw_return'],r['segment']))[:max(1,math.ceil(len(rows)/2))]])
        trajectories[task]=dict(dataset=str(p),sha256=sha(p),rows=5000,keys=keys,timeouts_field_present='timeouts' in keys,
            terminal_rows=np.flatnonzero(term).tolist(),observed_timeout_rows=np.flatnonzero(timeout).tolist(),
            discontinuity_rows=np.flatnonzero(jump).tolist(),inferred_reset_rows=np.flatnonzero(inferred).tolist(),
            max_abs_boundary_differences={str(i):float(diff[i]) for i in np.flatnonzero(jump)},
            observed_flags_only_segments=observed,reconstructed_segments=recovered,summaries=summaries,
            complete_count=len(complete),EOF_incomplete_segment=recovered[-1] if not recovered[-1]['complete_boundary'] else None,
            reconstruction_limit='timeouts field absent; inferred reset boundaries are evidence-based, not observed timeout flags; row0 is assumed dataset start, with no separately saved reset metadata')
        all_segments.extend(observed+recovered)
    write('TARGET_TRAJECTORIES.json',trajectories);csv_write('target_trajectories.csv',all_segments)
    print('task definition and target trajectory statistics complete',flush=True)

    # Resolve R11/R12 reuse explicitly and compute every T result from raw returns.
    t11=read(ROOT/'oracle_diag/results/coverage_oracle_20261016_r11_resume2/T_selected.json')
    t12=read(ROOT/'oracle_diag/results/coverage_oracle_20261018_r12/T_selected.json')
    assert [r['job']['out'] for r in t11['runs']]==[r['job']['out'] for r in t12['runs']]
    r13_receipts=read(ROOT/'oracle_diag/results/coverage_oracle_20261024_r13_finish1/reuse_receipts.json')['runs']
    r13_T=[r for r in r13_receipts if r['job']['arm']=='T']
    assert len(r13_T)==5
    groups=[('R11_Walker_T','walker2d-morph',[Path(r['job']['out']) for r in t11['runs']],False),
            ('R12_Walker_T_reuse_R11','walker2d-morph',[Path(r['job']['out']) for r in t12['runs']],True),
            ('R13_Walker_T','walker2d-morph',[Path(r['job']['out']) for r in r13_T],False),
            ('R14_Hopper_T','hopper-morph',[R14/'runs'/('hopp_w0_s%d'%s) for s in range(201,211)],False)]
    histories={};table=[];curve_rows=[];curve_stats={};runs_cached={}
    for label,task,paths,reused in groups:
        rr=[]
        for p in paths:
            if str(p) not in runs_cached:
                done=read(p/'completed.json');audit=read(p/'audit.json');cfg=read(p/'config.json')
                j=done['job'];assert j['arm']=='T' and j['temp']==10 and j['steps']==500000
                events=read(p/'evaluations.json')['events'];steps=list(range(10000,500001,10000))
                assert [e['step'] for e in events]==steps
                for f in (p/'tb').iterdir():
                    if f.is_file():track(f)
                tb=EventAccumulator(str(Path(short(p))/'tb'),size_guidance={'scalars':0}).Reload().Scalars('evaluation/target_score')
                assert [v.step for v in tb]==steps
                scores=[]
                for e,tb_value in zip(events,tb):
                    assert len(e['raw_returns'])==10
                    score=float(np.mean(normalize(task,np.asarray(e['raw_returns'],dtype=np.float64))))
                    assert abs(score-tb_value.value)<1e-4
                    scores.append(score)
                    if task=='hopper-morph':curve_rows.append(dict(seed=j['seed'],step=e['step'],score=score,episodes=10,path=str(p)))
                last5=float(np.mean(scores[-5:]));final=scores[-1]
                assert audit['passed'] and abs(audit['Last5']-last5)<1e-8 and abs(audit['Final']-final)<1e-8
                assert sha(p/'completed.json')==audit['completed_sha256'] and sha(p/'evaluations.json')==audit['evaluations_sha256']
                run=dict(seed=j['seed'],Last5=last5,Final=final,out=str(p),audit=str(p/'audit.json'),
                    audit_sha256=sha(p/'audit.json'),evaluations_sha256=sha(p/'evaluations.json'),learner=cfg.get('learner',cfg.get('config',{})))
                runs_cached[str(p)]=run
                if task=='hopper-morph':
                    rolling=np.convolve(scores,np.ones(5)/5,mode='valid');peak=int(np.argmax(rolling))+4
                    curve_stats[str(j['seed'])]=dict(Last5=last5,Final=final,maximum_single_point=max(scores),maximum_single_step=steps[int(np.argmax(scores))],
                        maximum_5point_mean=float(rolling.max()),maximum_5point_window=[steps[peak-4],steps[peak]],
                        final_minus_peak_5point=last5-float(rolling.max()),
                        block_means=[float(np.mean(scores[i:i+10])) for i in range(0,50,10)])
            rr.append(runs_cached[str(p)])
            table.append(dict(group=label,task=task,reused_from_R11=reused,seed=rr[-1]['seed'],Last5=rr[-1]['Last5'],Final=rr[-1]['Final'],path=str(p)))
        histories[label]=dict(task=task,reused=reused,seeds=[r['seed'] for r in rr],runs=rr,
            Last5=stats([r['Last5'] for r in rr],[r['seed'] for r in rr]),Final=stats([r['Final'] for r in rr],[r['seed'] for r in rr]))
    write('T_HISTORY.json',histories);csv_write('T_history_per_seed.csv',table)
    write('HOPPER_T_CURVE_DIAGNOSTICS.json',dict(seeds=curve_stats,steps=50,episodes_per_step=10,formal_change_point_test=False,
        interpretation='descriptive checkpoint/block/rolling summaries only; maxima are retrospective diagnostics, never model selection'))
    csv_write('hopper_T_curves.csv',curve_rows)
    print('historical T provenance and all Hopper curves independently recomputed',flush=True)

    # Inventory the entire project, not just file names that contain hopper.
    candidates=[];receipts_found=[];full_hopper=[];inventory=[]
    skip={'.git','__pycache__','node_modules','site-packages','.venv','Lib'}
    for current,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in skip and not (Path(current)/d).resolve().is_relative_to(OUT) ] if sys.version_info >= (3,9) else [d for d in dirs if d not in skip and str((Path(current)/d).resolve())!=str(OUT)]
        folder=Path(current)
        names=set(files)
        if {'next_state.npy','reward.npy','not_done.npy','raw_indices.npy'} <= names:
            headers={}
            for n in ('next_state.npy','reward.npy','not_done.npy','raw_indices.npy'):
                a=np.load(str(folder/n),mmap_mode='r',allow_pickle=False);headers[n]=dict(shape=list(a.shape),dtype=str(a.dtype))
            receipt=read(folder/'completed.json') if 'completed.json' in names else None
            r=dict(path=str(folder),headers=headers,receipt=receipt)
            candidates.append(r)
            if headers['next_state.npy']['shape']==[999998,11] and receipt and receipt.get('task')=='hopper-morph' and receipt.get('complete'):
                full_hopper.append(str(folder))
        for n in files:
            p=folder/n
            if n.endswith('.npz') and (('oracle' in n.lower() and 'hopper' in n.lower()) or ('hopper' in n.lower() and 'replay' in n.lower())):
                with np.load(str(p),allow_pickle=False) as z:headers={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}
                inventory.append(dict(path=str(p),sha256=sha(track(p)),headers=headers))
            if n.endswith('.json') and ('oracle' in str(folder).lower() or 'cache' in n.lower()) and n in ('completed.json','plan.json','acceptance.json'):
                try:
                    a=read(p)
                    if a.get('task')=='hopper-morph' or (a.get('n')==999998 and 'hopper' in str(p).lower()):receipts_found.append(dict(path=str(p),receipt=a))
                except (ValueError,UnicodeError):pass
    source_audit=read(ROOT/'oracle_diag/results/v3_20261008/stage_a2/hopper_source_acceptance.json')
    partial_audit=read(ROOT/'oracle_diag/results/v2_20261006/stage_c/hopper_morph_acceptance.json')
    cache_status=dict(search_root=str(ROOT),excluded_directories=sorted(skip),full_indexed_candidates=candidates,
        full_hopper_morph_cache_found=bool(full_hopper),full_hopper_cache_paths=full_hopper,other_hopper_receipts=receipts_found,npz_inventory=inventory,
        canonical_source_rows=999998,source_factual_sample_acceptance=dict(path='oracle_diag/results/v3_20261008/stage_a2/hopper_source_acceptance.json',n=source_audit['n'],accepted=source_audit['accepted']),
        historical_counterfactual_sample=dict(path=partial_audit['oracle_path'],n=partial_audit['n'],accepted=partial_audit['accepted']),
        conclusion='No complete accepted Hopper-morph source oracle cache found in searched project; historical 5000-row sample is not a full999998-row cache' if not full_hopper else 'Full-cache candidate found; requires review of acceptance before reuse',
        new_cache_generated=False,no_simulator_access=True)
    write('ORACLE_CACHE_INVENTORY.json',cache_status)
    # Source datasets are hashed without simulator access; data are never rewritten.
    source_file=track(Path.home()/'.d4rl/datasets/hopper_medium-v2.hdf5')
    with h5py.File(source_file,'r') as f:
        n=len(f['observations']);canonical=int((~np.asarray(f['timeouts'][:-1],dtype=bool)).sum())
    assert n==1000000 and canonical==999998
    # All input reads and old protected paths must still be unchanged.
    changes=[p for p,h in protected.items() if sha(p)!=h]
    changes += [p for p,h in INPUTS.items() if sha(p)!=h]
    assert not changes,changes[:5]
    write('INPUT_MANIFEST.json',dict(sha256=INPUTS,count=len(INPUTS)))
    write('READ_ONLY_VERIFICATION.json',dict(passed=True,CPU_only=True,old_protected_count=len(protected),read_input_count=len(INPUTS),changed=[],
        training_runs=0,optimization_updates=0,new_oracle_rows=0,simulator_steps=0,no_learner_or_simulator_import=True,
        CUDA_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],next_stage_started=False,
        elapsed_seconds=time.time()-begin,verified_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))
    print(json.dumps(dict(stage0_CPU_data_complete=True,read_input_count=len(INPUTS),old_protected=len(protected),elapsed=time.time()-begin)),flush=True)

if __name__=='__main__':main()
