"""V12 0.3 engineering utilities; no scientific imports at module import time."""
import datetime
import csv
import io
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify_package():
    f=read(HERE/'FROZEN.json')
    for name,h in f['files'].items():assert sha(HERE/name)==h,'Changed acceptance package: '+name
    return f


def verify_project(root,manifest_hash):
    root=Path(root).resolve()
    assert sha(root/'MANIFEST.json')==manifest_hash
    result={'MANIFEST.json':manifest_hash}
    for x in read(root/'MANIFEST.json')['files']:
        p=(root/x['path']).resolve()
        assert root in p.parents and p.is_file(), 'Missing/external transfer file: '+x['path']
        assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'], 'Changed transfer input: '+x['path']
        result[x['path']]=x['sha256']
    return result


def hash_tree(root):
    root=Path(root).resolve()
    result={}
    for p in sorted(root.rglob('*')):
        if p.is_file():
            assert root in p.resolve().parents, 'External symlink in protected directory'
            result[p.relative_to(root).as_posix()]=sha(p)
    return result


def gpu_info(gpu):
    fields='index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu'
    p=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu='+fields,'--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=20,check=True)
    rows=[s.strip() for s in p.stdout.splitlines() if s.strip()]
    assert len(rows)==1,'Select exactly one allocated physical GPU'
    v=[x.strip() for x in rows[0].split(',')]
    assert len(v)==7 and v[1].startswith('GPU-') and 'A6000' in v[2], 'Expected one RTX A6000'
    return dict(index=int(v[0]),uuid=v[1],name=v[2],driver=v[3],memory_used_MiB=int(v[4]),
                memory_total_MiB=int(v[5]),utilization_gpu_percent=int(v[6]),raw=p.stdout)


def compute_process_rows(text):
    """Parse all visible compute processes; retain UUIDs instead of blocking on other GPUs."""
    result=[]
    for row in csv.reader(io.StringIO(text)):
        if not row or not any(x.strip() for x in row):continue
        assert len(row)==4,'Unexpected compute-process query format'
        uuid,pid,name,memory=[x.strip() for x in row]
        assert uuid.startswith('GPU-') and pid.isdigit(),'Unidentifiable compute-process record'
        result.append(dict(gpu_uuid=uuid,pid=int(pid),process_name=name,memory_raw=memory))
    return result


def pmon_process_rows(text,index):
    result=[];observed=0
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        parts=line.split()
        assert len(parts)>=10 and parts[0].isdigit(),'Unsupported/unidentifiable pmon output'
        assert int(parts[0])==index,'pmon did not report the selected device'
        observed+=1
        if parts[1]=='-':
            assert all(x=='-' for x in parts[1:]),'Ambiguous empty pmon row'
        else:
            assert parts[1].isdigit(),'Unidentifiable pmon PID'
            result.append(dict(index=index,pid=int(parts[1]),kind=parts[2],raw=line))
    assert observed>0,'pmon did not provide a usable selected-GPU record'
    return result


def admission_reasons(sample,policy):
    v=sample['gpu'];reasons=[]
    if v['uuid']!=policy['expected_uuid']:reasons.append('GPU UUID differs from the specifically diagnosed device')
    if v['driver']!=policy['observed_driver']:reasons.append('Driver differs from the diagnosed resource snapshot')
    if not 0<=v['memory_used_MiB']<=policy['max_resident_MiB']:reasons.append('Memory exceeds the diagnosed resident-memory allowance')
    if v['utilization_gpu_percent']!=0:reasons.append('GPU utilization is nonzero')
    for key in ('selected_compute_processes','selected_pmon_processes'):
        if not isinstance(sample[key],list):reasons.append('Process visibility is incomplete: '+key)
        elif sample[key]:reasons.append('An active process is visible on the selected GPU: '+key)
    return reasons


def nvidia_read(args):
    p=subprocess.run(['nvidia-smi']+args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     text=True,timeout=20,check=True)
    return dict(stdout=p.stdout,stderr=p.stderr)


def idle_gpu(gpu):
    policy=read(HERE/'config_template.json')['resource_admission']
    samples=[];uuid=None
    for i in range(policy['consecutive_samples']):
        value=gpu_info(gpu)
        if uuid is None:uuid=value['uuid']
        assert uuid==value['uuid'],'Selected GPU identity changed between samples'
        compute=nvidia_read(['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                             '--format=csv,noheader,nounits'])
        pmon=nvidia_read(['pmon','-i',str(value['index']),'-c','1'])
        all_processes=compute_process_rows(compute['stdout'])
        selected=[p for p in all_processes if p['gpu_uuid']==value['uuid']]
        sample=dict(gpu=value,time=utc(),selected_compute_processes=selected,
                    selected_pmon_processes=pmon_process_rows(pmon['stdout'],value['index']),
                    all_compute_processes=all_processes,raw_compute_query=compute,raw_pmon=pmon)
        sample['rejection_reasons']=admission_reasons(sample,policy)
        samples.append(sample)
        assert not sample['rejection_reasons'],('Selected GPU admission blocked; no retry/switch/signals',samples)
        if i+1<policy['consecutive_samples']:time.sleep(policy['sample_interval_seconds'])
    result=dict(samples[-1]['gpu'])
    result.update(resource_admission_policy=policy,resource_admission_samples=samples)
    return result


def read_freeze_json(path):
    def unique_pairs(pairs):
        d={}
        for key,value in pairs:
            assert key not in d,'Duplicate key in freeze JSON'
            d[key]=value
        return d
    def invalid_constant(value):raise ValueError('Non-finite JSON value: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=unique_pairs,parse_constant=invalid_constant)


def freeze_semantic_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def verify_freeze_content(actual_path,expected_path):
    actual,expected=read_freeze_json(actual_path),read_freeze_json(expected_path)
    assert freeze_semantic_digest(actual)==freeze_semantic_digest(expected), 'Freeze JSON contents differ'
    return dict(parsed_JSON_content_equal=True,actual_raw_sha256=sha(actual_path),
        expected_raw_sha256=sha(expected_path),canonical_content_sha256=freeze_semantic_digest(actual),
        raw_bytes_equal=sha(actual_path)==sha(expected_path))


def verify_reference_source(code_root,expected_path):
    code_root=Path(code_root).resolve()
    result=verify_freeze_content(code_root/'FROZEN.json',expected_path)
    expected=read_freeze_json(expected_path)
    for relative,h in expected['files'].items():
        p=(code_root/relative).resolve()
        assert code_root in p.parents and p.is_file(),'Missing/external frozen source'
        assert sha(p)==h,'Frozen source bytes changed: '+relative
    result.update(all_referenced_source_file_bytes_verified=True,source_files=len(expected['files']))
    return result


def full_smi():
    p=subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=20,check=True)
    return p.stdout


def existing_environment_snapshot():
    prefix=Path(sys.prefix).resolve()
    mu=Path(os.environ['MUJOCO_PY_MUJOCO_PATH']).resolve()
    condadir=prefix/'conda-meta'
    binding=Path(metadata.distribution('mujoco-py').locate_file('mujoco_py')).resolve()
    assert prefix in binding.parents
    return dict(prefix=str(prefix),python=sys.version,executable=sys.executable,
        pip_packages={x.metadata['Name']:x.version for x in metadata.distributions()},
        conda_packages=[read(p) for p in sorted(condadir.glob('*.json'))],
        conda_metadata_hashes=hash_tree(condadir),original_binding=str(binding),
        original_binding_hashes=hash_tree(binding),mujoco_library=str(mu/'bin/libmujoco210.so'),
        mujoco_library_sha256=sha(mu/'bin/libmujoco210.so'))


def check_environment_protection(snapshot):
    assert hash_tree(Path(snapshot['prefix'])/'conda-meta')==snapshot['conda_metadata_hashes'], 'Conda metadata changed'
    assert hash_tree(snapshot['original_binding'])==snapshot['original_binding_hashes'], 'Installed MuJoCo binding changed'
    assert sha(snapshot['mujoco_library'])==snapshot['mujoco_library_sha256'], 'MuJoCo native library changed'


def private_binding(out):
    original=Path(metadata.distribution('mujoco-py').locate_file('mujoco_py')).resolve()
    assert Path(sys.prefix).resolve() in original.parents
    vendor=out/'private_runtime'
    vendor.mkdir()
    target=vendor/'mujoco_py'
    shutil.copytree(original,target)
    sys.path.insert(0,str(vendor))
    write(out/'PRIVATE_BINDING_COPY.json',dict(original=str(original),copy=str(target),
        input_hashes=hash_tree(original),copied_bytes=sum(p.stat().st_size for p in target.rglob('*') if p.is_file())))
    return target


def length_stats(values):
    import numpy as np
    x=np.asarray(values,dtype=np.int64)
    assert len(x) and ((x>=1)&(x<=1000)).all()
    return dict(n=int(x.size),mean=float(x.mean()),median=float(np.median(x)),minimum=int(x.min()),
                fraction_under_200=float((x<200).mean()),episodes_under_200=int((x<200).sum()))


def loss_stats(values):
    import numpy as np
    x=np.asarray(values,dtype=np.float64)
    assert len(x) and np.isfinite(x).all()
    return dict(mean=float(x.mean()),minimum=float(x.min()),median=float(np.median(x)),
                p99=float(np.percentile(x,99)),maximum=float(x.max()),first=float(x[0]),last=float(x[-1]))


def verify_stage02(prior):
    prior=Path(prior).resolve()
    assert prior==(Path.home()/'v12_linux/outputs/stage0_2_20261107_json_repair2').resolve()
    c=read(prior/'STAGE0_2_COMPLETED.json')
    assert c['completed'] and c['jobs_completed']==c['jobs_CPU_audited']==3
    assert c['learner_updates']==40000 and c['Adam_updates']==120000
    assert c['deterministic_comparison_passed'] and c['full_entry_prefix_equal']
    assert c['policy_evaluation_episodes']==20 and c['physical_gpu_uuid']==read(HERE/'config_template.json')['resource_admission']['expected_uuid']
    for relative,h in c['evidence_sha256'].items():
        p=(prior/relative).resolve()
        assert prior in p.parents and sha(p)==h,'Prior 0.2 evidence changed: '+relative
    assert read(prior/'INPUT_PROTECTION_AFTER.json')['passed']
    for name,steps in (('determinism_a',10000),('determinism_b',10000),('entry_20k',20000)):
        a=read(prior/name/'AUDIT.json');v=read(prior/name/'COMPLETED.json')
        assert a['passed'] and a['step']==steps and v['counts']['learner_updates']==steps
    code=Path.home()/'v12_linux/stage0_2_json_repair_20261107'
    source=verify_reference_source(code,HERE/'references/STAGE0_2_SOURCE_FREEZE.json')
    freeze=verify_freeze_content(prior/'CODE_FREEZE.json',HERE/'references/STAGE0_2_SOURCE_FREEZE.json')
    return dict(passed=True,receipt_sha256=sha(prior/'STAGE0_2_COMPLETED.json'),
        referenced_evidence_files=len(c['evidence_sha256']),source=source,freeze=freeze,
        source_code_path=str(code),user_authorization='2026-11-07: 启动0.3')


def publish(path,value):
    """Atomic, no-overwrite publication for cross-process barriers."""
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('x',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.link(str(tmp),str(path))
    finally:
        if tmp.exists():tmp.unlink()


