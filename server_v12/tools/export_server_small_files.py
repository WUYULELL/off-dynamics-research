"""Read-only server artifact export for the requested V12/V13 archive.

No training, model loads, environment steps, GPU calls, SSH or installations.
Only known report/config/receipt/table names are read. New output is created in
the user's home; existing experiment outputs are neither changed nor deleted.
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
 'v12_linux/outputs/stage2_root_20261108_continue1',
 'v12_linux/outputs/stage3_switch_20261110',
 'v13_action_fix/outputs/stage0_20261110',
 'v13_action_fix/outputs/stage0_20261110_report_repaired',
 'v13_action_fix/outputs/stage1_20261110_shared60',
]
NAMES={'REPORT_ZH.md','STAGE3_REPORT_ZH.md','RESULT.json','config.json','COMPLETED.json',
 'STAGE0_1_COMPLETED.json','STAGE0_2_COMPLETED.json','STAGE0_3_COMPLETED.json',
 'STAGE0_COMPLETED.json','STAGE1_COMPLETED.json','STAGE3_COMPLETED.json',
 'AUDIT.json','AUTHORIZATION.json','PLAN.json','PROTOCOL.json','protocol.json',
 'TRAJECTORY_SUBSETS.json','ROOT_CONFIG.json','ROOT_DATA_ENV_COMPARISON.json',
 'WEIGHT_STATISTICS.json','ROOT_IMPLEMENTATION_AUDIT.json','SOURCE_HASHES.json',
 'INPUT_PROTECTION_AFTER.json','TRAINING_COMPLETED.json','WEIGHTS_COMPLETED.json'}


def h(data):return hashlib.sha256(data).hexdigest()


def redact(text):
    changes=[]
    for kind,pattern in [('server_user',r'(/mnt/cs_dsns_public/<USER>/\s\"\'<>]+)'),
                         ('home_user',r'(/home/<USER>/\s\"\'<>]+)'),
                         ('windows_user',r'([A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/))([^\\/\s\"<>]+)')]:
        text,n=re.subn(pattern,r'\1<USER>',text)
        if n:changes.append(dict(kind=kind,count=n,replacement='<USER>'))
    return text,changes


def sensitive(text):
    patterns={
      'server_login_identifier':r'\becs-instance-[\w-]+\b|\bcs_dsns_(?!public\b)[A-Za-z0-9_]+\b',
      'private_key':r'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|ssh-(?:rsa|ed25519)\s+[A-Za-z0-9+/]{30,}',
      'token':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,})\b',
      'credential_literal':r'(?i)(?:password|passwd|access_token|api_key)\s*[\"\']?\s*[:=]\s*[\"\'][^\"\']{5,}[\"\']',
      'other_process_identifiers':r'(?is)[\"\'](?:other_pids|visible_pids|other_GPU_pids)[\"\']\s*:\s*\[\s*\d|GPU-[a-fA-F0-9-]{20,}\s*,\s*\d{3,}\s*,',
    }
    found=[name for name,pattern in patterns.items() if re.search(pattern,text)]
    for m in re.finditer(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',text):
        if m.group()!='2.1.2.14' and all(int(x)<=255 for x in m.group().split('.')):
            found.append('IPv4_address');break
    return found


def main():
    home=Path.home()
    tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output=home/('github_small_export_'+tag)
    output.mkdir(exist_ok=False)
    selected=set();missing=[]
    for relative in STAGES:
        root=home/relative
        if not root.is_dir():missing.append(relative);continue
        for p in root.iterdir():
            if p.is_file() and (p.name in NAMES or p.suffix=='.csv'):
                selected.add(p)
        for name in ('config.json','COMPLETED.json','AUDIT.json'):
            selected.update(p for p in (root/'runs').glob('*/'+name) if p.is_file())
    cache=home/'v13_action_fix/outputs/stage1_20261110/runs/full_cache'
    for name in ('CACHE_ACCEPTANCE.json','CACHE_STATISTICS.json','CACHE_REPLAY_CHECK.json'):
        if (cache/name).is_file():selected.add(cache/name)
    copied=[];excluded=[]
    for source in sorted(selected):
        rel=source.relative_to(home).as_posix()
        raw=source.read_bytes();sha=h(raw)
        meta=dict(source='${HOME}/'+rel,bytes=len(raw),source_sha256=sha)
        if len(raw)>LIMIT:
            excluded.append(dict(meta,reason='over_5_MB; retain original; export small tables separately'));continue
        try:text=raw.decode('utf-8')
        except UnicodeDecodeError:
            excluded.append(dict(meta,reason='non_UTF8_or_binary'));continue
        clean,changes=redact(text)
        reasons=sensitive(clean)
        if reasons:
            excluded.append(dict(meta,reason='sensitive_information_detected',finding_types=reasons));continue
        target=output/rel;target.parent.mkdir(parents=True,exist_ok=True)
        data=clean.encode('utf-8');target.write_bytes(data)
        assert h(source.read_bytes())==sha,'Source changed during read-only export'
        copied.append(dict(meta,snapshot=rel,snapshot_sha256=h(data),transformations=changes,exact_byte_copy=(raw==data)))
    manifest=dict(files=copied,excluded=excluded,missing_directories=missing,
        source_path_template='${HOME} resolves to original user home; actual account name is not exported',
        training_updates=0,environment_steps=0,GPU_calls=0,original_files_changed=False)
    (output/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(output))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('New export ZIP:',archive)
    print('SHA256:',h(archive.read_bytes()))
    print('Copied:',len(copied),'Excluded:',len(excluded),'Missing directories:',len(missing))
    print('No training/environment/GPU operations; originals retained unchanged.')


if __name__=='__main__':main()
