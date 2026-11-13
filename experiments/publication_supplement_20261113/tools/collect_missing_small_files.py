"""Collect V12/V13/V14 small artifacts. Standard library, no experiment imports.

Only new export files are written. Original outputs stay untouched. No GPU,
training, model loads, simulator, network or process inspection is performed.
"""
import datetime
import hashlib
import json
from pathlib import Path
import re
import zipfile

LIMIT=5_000_000
STAGES=[
 'v12_linux/outputs/stage0_1_20261106',
 'v12_linux/outputs/stage0_2_20261107_json_repair2',
 'v12_linux/outputs/stage0_3_20261107',
 'v12_linux/outputs/stage1A_20261107',
 'v12_linux/outputs/stage1B_20261107',
 'v12_linux/outputs/stage2_root_20261107',
 'v12_linux/outputs/stage2_root_20261108_continue1',
 'v12_linux/outputs/stage3_shared60_20261108',
 'v12_linux/outputs/stage3_switch_20261110',
 'v13_action_fix/outputs/stage0_20261110',
 'v13_action_fix/outputs/stage0_20261110_report_repaired',
 'v13_action_fix/outputs/stage1_20261110',
 'v13_action_fix/outputs/stage1_20261110_shared60',
 'v14_diag/outputs/diagnostics_20261112_215443',
]
SUBFILES={'config.json','completed.json','audit.json','result.json','report_zh.md',
 'measured.json','tier_completed.json','b0_result.json','a2_log_check.json',
 'cache_acceptance.json','cache_statistics.json','cache_replay_check.json'}
SKIP_DIRS={'private_runtime','tb','logs','__pycache__','raw','upstream_root','environment'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def sanitize(text):
    changes=[]
    for name,pattern in [('server_user',r'(/mnt/cs_dsns_public/)([^/\s\"\'<>]+)'),
                         ('home_user',r'(/home/)([^/\s\"\'<>]+)'),
                         ('windows_user',r'([A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/))([^\\/\s\"<>]+)')]:
        text,n=re.subn(pattern,r'\1<USER>',text)
        if n:changes.append(dict(kind=name,count=n,replacement='<USER>'))
    return text,changes


def private_findings(text):
    patterns={
      'login_identifier':r'\becs-instance-[\w-]+\b|\bcs_dsns_(?!public\b)[A-Za-z0-9_]+\b',
      'key':r'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|ssh-(?:rsa|ed25519)\s+[A-Za-z0-9+/]{30,}',
      'token':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,})\b',
      'password':r'(?i)(?:password|passwd|access_token|api_key)\s*[\"\']?\s*[:=]\s*[\"\'][^\"\']{5,}[\"\']',
      'other_process_data':r'(?is)[\"\'](?:other_pids|visible_pids|other_GPU_pids)[\"\']\s*:\s*\[\s*\d|GPU-[a-fA-F0-9-]{20,}\s*,\s*\d{3,}\s*,',
    }
    found=[k for k,v in patterns.items() if re.search(v,text)]
    for m in re.finditer(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',text):
        line=text[text.rfind('\n',0,m.start())+1:text.find('\n',m.end()) if '\n' in text[m.end():] else len(text)]
        is_requirement=re.fullmatch(r'\s*[A-Za-z0-9_.\[\],-]+==[A-Za-z0-9_.+!-]+\s*',line)
        if m.group()!='2.1.2.14' and not is_requirement and all(int(x)<=255 for x in m.group().split('.')):
            found.append('IPv4');break
    return found


def selected_files(root):
    for p in root.rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(root)
        if any(s in SKIP_DIRS for s in relative.parts[:-1]):continue
        if p.suffix.lower() not in ('.json','.md','.csv','.tsv','.txt'):continue
        # Never export raw episode logs, loss streams, readiness/PID launch or resource samples.
        if re.search(r'(?i)(?:^eval_\d|(?:^|_)raw_loss|^loss(?:es|_log|_stream)?[._]|^steps[._]|launch|gpu_blocked|gpu_admission|resource|started|heartbeat|status)',p.name):continue
        if len(relative.parts)==1:
            if p.suffix.lower()=='.txt' and not re.search(r'(?i)(pip|conda|requirements)',p.name):continue
            yield p
        elif p.name.lower() in SUBFILES or p.suffix.lower() in ('.csv','.tsv'):
            yield p
        elif relative.parts[0] in ('A','B','C') and p.suffix.lower()=='.json':
            # Aggregate diagnostic results and eight B0 trajectory summaries;
            # per-step trajectory arrays are .npz and are never selected.
            yield p


def main():
    home=Path.home()
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out=home/('github_missing_export_'+stamp)
    out.mkdir(exist_ok=False)
    stages=list(STAGES)
    latest=home/'v14_diag/outputs/LATEST_RUN.txt'
    if latest.is_file():
        p=Path(latest.read_text(encoding='utf-8').strip()).resolve()
        allowed=(home/'v14_diag/outputs').resolve()
        if allowed in p.parents:
            name=p.relative_to(home.resolve()).as_posix()
            if name not in stages:stages.append(name)
    paths=set();missing=[]
    for name in stages:
        root=home/name
        if not root.is_dir():missing.append(name);continue
        paths.update(selected_files(root))
    copied=[];excluded=[]
    for source in sorted(paths):
        assert home.resolve() in source.resolve().parents,'Refusing external symlink'
        rel=source.relative_to(home).as_posix();size=source.stat().st_size;before=sha(source)
        meta=dict(source='${HOME}/'+rel,bytes=size,source_sha256=before)
        if size>LIMIT:
            excluded.append(dict(meta,reason='over_5_MB; retained on server, requires separate small table export'));continue
        try:text=source.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            excluded.append(dict(meta,reason='not_UTF8'));continue
        # Read bytes to retain original LF/CRLF and BOM whenever no replacement occurs.
        raw=source.read_bytes();text=raw.decode('utf-8')
        clean,changes=sanitize(text);findings=private_findings(clean)
        if findings:
            excluded.append(dict(meta,reason='sensitive_information',finding_types=findings));continue
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
        data=clean.encode('utf-8');target.write_bytes(data)
        assert sha(source)==before,'Original file changed during export'
        copied.append(dict(meta,snapshot=rel,snapshot_sha256=hashlib.sha256(data).hexdigest(),
            transformations=changes,exact_byte_copy=(raw==data)))
    manifest=dict(files=copied,excluded=excluded,missing_directories=missing,
        original_files_changed=False,training_updates=0,environment_steps=0,GPU_calls=0,
        note='Only reports/configs/receipts/small tables. Oversized original results remain at their paths; no model/data/cache/loss/trajectory arrays copied.')
    (out/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('ZIP:',archive)
    print('SHA256:',sha(archive))
    print('Included:',len(copied),'Excluded:',len(excluded),'Missing directories:',len(missing))
    print('No training/environment/GPU operations. Original outputs unchanged.')


if __name__=='__main__':main()
