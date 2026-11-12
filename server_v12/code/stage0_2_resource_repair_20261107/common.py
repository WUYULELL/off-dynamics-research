"""V12 0.2 engineering utilities. No imports of Torch, Gym, or MuJoCo."""
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


def verify_failed_start(old):
    old=Path(old).resolve();cfg=read(HERE/'config_template.json')
    assert old==(Path.home()/'v12_linux/outputs/stage0_2_20261106').resolve()
    failed=read(old/'FAILED.json')
    assert failed['completed_audited_jobs']==[]
    assert failed['input_protection'].get('transfer_unchanged') is True
    assert failed['input_protection'].get('environment_unchanged') is True
    assert 'gpu=idle_gpu(args.gpu)' in failed['traceback']
    assert 'Selected GPU is occupied; stop without terminating/switching devices' in failed['traceback']
    assert cfg['resource_admission']['expected_uuid'] in failed['traceback']
    assert "'memory_used_MiB': 3" in failed['traceback']
    assert not list(old.glob('*.job.json')) and not (old/'WORKER_AUTHORIZATION.json').exists()
    assert not (old/'PRE_RESULT_COMMITMENT.json').exists()
    assert not any((old/n).exists() for n in ('determinism_a','determinism_b','entry_20k'))
    assert sha(old/'CODE_FREEZE.json')==cfg['original_code_freeze_sha256']
    hashes=hash_tree(old)
    return dict(passed=True,old_failure=str(old),old_failure_sha256=sha(old/'FAILED.json'),
        old_files_sha256=hashes,verified_training_updates_before_failure=0,
        verified_environment_steps_before_failure=0,
        reason='Original controller stopped at the first resource gate before creating job files or workers')


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


def verify_prior_receipt(prior):
    prior=Path(prior).resolve()
    receipt=read(prior/'STAGE0_1_COMPLETED.json')
    assert receipt['completed'] and receipt['historical_technical_checks']
    assert receipt['new_training_updates']==0 and receipt['GPU_tests']==0
    expected=read(HERE/'config_template.json')['approved_stage0_1_report_sha256']
    assert sha(prior/'REPORT_ZH.md')==expected, 'Not the user-approved 0.1 report'
    for rel,h in receipt['evidence_sha256'].items():
        p=(prior/rel).resolve()
        assert prior in p.parents and sha(p)==h, 'Prior 0.1 evidence changed: '+rel
    return dict(receipt_sha256=sha(prior/'STAGE0_1_COMPLETED.json'),
                report_sha256=expected,files_checked=len(receipt['evidence_sha256']),
                user_approved=True,approval='2026-11-06: 0.1 通过，继续 0.2')


def compare_loss_arrays(a,b):
    import numpy as np
    assert a.dtype==b.dtype==np.float32 and a.shape==b.shape
    assert a.ndim==2 and a.shape[1]==3
    assert np.isfinite(a).all() and np.isfinite(b).all(), 'Non-finite loss array'
    bits_a,bits_b=a.view(np.uint32),b.view(np.uint32)
    bad=np.any(bits_a!=bits_b,axis=1)
    first=None
    if bad.any():
        i=int(np.flatnonzero(bad)[0]);j=int(np.flatnonzero(bits_a[i]!=bits_b[i])[0])
        first=dict(step=i+1,field=('V','Q','actor')[j],a=float(a[i,j]),b=float(b[i,j]),
                   a_bits=int(bits_a[i,j]),b_bits=int(bits_b[i,j]))
    return dict(bitwise_equal=not bool(bad.any()),records_compared=len(a),
                records_with_difference=int(bad.sum()),first_difference=first)


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
