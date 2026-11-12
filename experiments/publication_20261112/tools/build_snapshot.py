"""Prepare additive public snapshots; never modifies original experiment files."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
DEST=WORK/'payload_v4'
MAX_BYTES=5_000_000
ENTRIES=[]
EXCLUDED=[]
GAPS=[]
COUNTS={}


def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def redact(text):
    changes=[]
    rules=[('server_username_in_absolute_path',r'(/mnt/cs_dsns_public/<USER>/\s\"\'<>]+)',r'\1<USER>'),
           ('home_username_in_absolute_path',r'(/home/<USER>/\s\"\'<>]+)',r'\1<USER>'),
           ('windows_username_in_absolute_path',r'([A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/))([^\\/\s\"<>]+)',r'\1<USER>')]
    for kind,pattern,replacement in rules:
        text,n=re.subn(pattern,replacement,text)
        if n:changes.append(dict(kind=kind,count=n,replacement='<USER>'))
    return text,changes


def public_origin(path):
    s=str(path).replace('\\','/')
    r=str(ROOT).replace('\\','/')
    if s.startswith(r+'/'):return '${PROJECT_ROOT}/'+s[len(r)+1:]
    return redact(s)[0]


def findings(text):
    patterns={
      'server_hostname':r'\becs-instance-[A-Za-z0-9-]+\b',
      'server_account_outside_redacted_path':r'\bcs_dsns_(?!public\b)[A-Za-z0-9_]+\b',
      'IPv4_address':r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',
      'private_key':r'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|\bssh-(?:rsa|ed25519)\s+[A-Za-z0-9+/]{30,}',
      'access_token':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})\b',
      'credential_literal':r'(?i)(?:password|passwd|access_token|api_key)\s*[\"\']?\s*[:=]\s*[\"\'][^\"\']{5,}[\"\']',
      'password_like_literal':r'\b[A-Za-z][A-Za-z0-9]{3,}@\d{3,}\b',
      'runtime_other_process_ids':r'(?is)[\"\'](?:other_pids|visible_pids|other_GPU_pids)[\"\']\s*:\s*\[\s*\d',
      'raw_GPU_process_snapshot':r'GPU-[a-fA-F0-9-]{20,}\s*,\s*\d{3,}\s*,',
    }
    found=[]
    for k,v in patterns.items():
        matches=list(re.finditer(v,text))
        if k=='IPv4_address':
            valid=[]
            for m in matches:
                line=text[text.rfind('\n',0,m.start())+1:text.find('\n',m.end()) if '\n' in text[m.end():] else len(text)]
                requirement=bool(re.fullmatch(r'\s*[A-Za-z0-9_.\[\],-]+==[A-Za-z0-9_.+!-]+\s*',line))
                if m.group()!='2.1.2.14' and not requirement and all(int(x)<=255 for x in m.group().split('.')):valid.append(m)
            matches=valid
        if matches:found.append(dict(kind=k,lines=sorted({text.count('\n',0,m.start())+1 for m in matches})[:20]))
    return found


def exclude(source,data,reason,**extra):
    EXCLUDED.append(dict(source=source,bytes=len(data),sha256=digest(data),reason=reason,status='未上传，保留在本机',**extra))


def add(data,target,source,kind='exact_copy',**meta):
    assert not (DEST/target).exists(),target
    if len(data)>MAX_BYTES:
        exclude(source,data,'exceeds_5_MB_upload_limit',requested_snapshot=target)
        return
    ext=Path(target).suffix.lower()
    changes=[]
    if ext=='.docx':
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            texts='\n'.join(z.read(n).decode('utf-8',errors='replace') for n in z.namelist() if n.endswith('.xml'))
        clean,changes=redact(texts)
        problems=findings(clean)
        if changes or problems:
            exclude(source,data,'binary_editorial_document_contains_identifying_content; not altered',findings=problems,path_replacement_needed=bool(changes))
            return
    else:
        try:text=data.decode('utf-8')
        except UnicodeDecodeError:
            exclude(source,data,'binary_or_non_UTF8_not_in_small_text_snapshot_scope');return
        clean,changes=redact(text)
        problems=findings(clean)
        if problems:
            exclude(source,data,'public_sensitive_information_detected',findings=problems,requested_snapshot=target)
            return
        data_out=clean.encode('utf-8')
        if not changes:assert data_out==data
        else:kind=kind+'; absolute_path_username_replacement'
        data_original=data
        data=data_out
    original=locals().get('data_original',data)
    p=DEST/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    ENTRIES.append(dict(source=source,snapshot=target,source_bytes=len(original),source_sha256=digest(original),
        snapshot_bytes=len(data),snapshot_sha256=digest(data),exact_byte_copy=(original==data),
        kind=kind,transformations=changes,**meta))


def file(source,target,**meta):
    p=ROOT/source
    add(p.read_bytes(),target,public_origin(p),**meta)


def archive(source,top,historical=None):
    p=ROOT/source
    data=p.read_bytes();archive_hash=digest(data)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        for item in z.infolist():
            if item.is_dir():continue
            rel=item.filename.split('/',1)[1]
            target=f'{top}/'+('historical_code/'+historical+'/' if historical else 'code/')+rel
            origin=public_origin(p)+'!/'+item.filename
            body=z.read(item)
            if '/upstream_root/' in '/'+rel:
                exclude(origin,body,'upstream_ROOT_tree_omitted_by_request; cite pinned source commit');continue
            if Path(rel).suffix.lower() in ('.npy','.npz','.pt','.pth','.hdf5','.h5','.pkl','.png','.jpg','.jpeg','.so','.dll'):
                exclude(origin,body,'dataset_model_cache_binary_not_authorized');continue
            add(body,target,origin,source_archive_sha256=archive_hash,source_archive_bytes=len(data),source_member=item.filename)


def parse_chat(id,top,stage,expected_report_sha=None):
    p=Path('C:/Users/<USER>/.codex/attachments')/id/'已粘贴的文本.txt'
    raw=p.read_bytes();text=raw.decode('utf-8-sig').replace('\r\n','\n')
    exclude(f'chat_attachment:{id}/已粘贴的文本.txt',raw,
        'raw_terminal_transcript_contains_server_login_identifiers; not uploaded; requested report/JSON blocks extracted separately')
    marker=re.search(r'^# V1[23] ',text,re.M)
    if not marker:return
    tail=text[marker.start():]
    stop=re.search(r'^\S+@[^\n]*\$|^\S+@[^\n]*:~',tail,re.M)
    if stop:tail=tail[:stop.start()]
    body=tail.rstrip('\r\n')+'\n'
    body_bytes=body.encode('utf-8')
    verified=expected_report_sha is not None and digest(body_bytes)==expected_report_sha
    if expected_report_sha:assert verified,(id,digest(body_bytes),expected_report_sha)
    add(body_bytes,f'{top}/received_reports/{stage}/REPORT_FROM_CHAT.md',f'chat_attachment:{id}/已粘贴的文本.txt#report',
        kind='terminal_report_block_extraction; CRLF-to-LF reconstruction',attachment_sha256=digest(raw),
        server_original_report_sha256=expected_report_sha,server_original_report_hash_matched=verified,
        caveat='Original server file not downloaded; report block extracted from user-provided terminal paste. Original byte hash verified only when explicitly recorded.')
    for number,m in enumerate(re.finditer(r'^\{',text,re.M)):
        try:o,n=json.JSONDecoder().raw_decode(text[m.start():])
        except ValueError:continue
        if not isinstance(o,dict):continue
        if o.get('completed') is True or o.get('passed') is True:
            block=(text[m.start():m.start()+n]+'\n').encode('utf-8')
            name='COMPLETION_FROM_CHAT.json' if o.get('completed') else f'CHECK_FROM_CHAT_{number}.json'
            add(block,f'{top}/received_reports/{stage}/{name}',f'chat_attachment:{id}/已粘贴的文本.txt#JSON-{number}',
                kind='terminal_JSON_block_extraction',attachment_sha256=digest(raw),
                caveat='Faithful exported JSON block; original server-file byte hash is not available for this receipt.')
    # Tables are directly exported from the report block; no numeric inference.
    lines=body.splitlines();tables=[];current=[]
    for line in lines+['']:
        if line.startswith('|'):current.append(line)
        elif current:
            if len(current)>2:tables.append(current)
            current=[]
    for number,table in enumerate(tables,1):
        rows=[[v.strip() for v in line.strip('|').split('|')] for line in [table[0]]+table[2:]]
        f=io.StringIO(newline='');csv.writer(f,lineterminator='\n').writerows(rows)
        add(f.getvalue().encode('utf-8'),f'{top}/received_reports/{stage}/table_{number:02d}.csv',f'chat_attachment:{id}/已粘贴的文本.txt#report-table-{number}',
            kind='CSV_table_export_from_report',report_extracted_sha256=digest(body_bytes),attachment_sha256=digest(raw))


def main():
    DEST.mkdir(exist_ok=False)
    # Windows formal delivery and read-only diagnostics.
    directory=ROOT/'outputs/v11_saall_report_20261104'
    requested=['authorization.json','config.json','CORRECTION_FIELD_AUDIT.json','DIAGNOSTIC_GROUP_SUMMARY.json',
        'REPORT_REPAIR_COMMITMENT.json','REPORT_CODE_FROZEN.json','training_PRE_RESULT_COMMITMENT.json',
        'stage1/STAGE1_REPORT_ZH.md','stage1/STAGE1_COMPLETED.json','stage1/ANALYSIS_COMPLETED.json',
        'stage1/ANALYSIS_BASE_COMPLETED.json','stage1/DELIVERY_VERIFIED.json','stage1/EXECUTION_SUMMARY.json',
        'stage1/protocol.json','stage1/training_plan.json','stage1/reference_receipts.json']
    for name in requested:
        file(str((directory/name).relative_to(ROOT)),f'experiments/v11/stage1_20261104/{name}')
    for name in ['SEED207_PREFIX_COMPARISON.json','SEED207_VECTOR_AUDIT.json','INDEPENDENT_VECTOR_VERIFIED.json','independent_vector_verify.py']:
        file(str((directory/name).relative_to(ROOT)),f'experiments/v11/seed207_20261104/{name}')
    for p in (ROOT/'oracle_diag/v11_report_fix_20261104').glob('*.py'):
        file(str(p.relative_to(ROOT)),f'experiments/v11/stage1_20261104/report_code/{p.name}')
    for p in (ROOT/'outputs/v11_episode_length_diagnostic_20261104').iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.json','.csv'):
            file(str(p.relative_to(ROOT)),f'experiments/v11/episode_lengths_20261104/{p.name}')
    full=directory/'stage1/STAGE1_RESULT.json'
    exclude(public_origin(full),full.read_bytes(),'exceeds_5_MB; user explicitly approved table-only export; 未上传，保留在本机')
    for p in (WORK/'derived_v11_tables_v2').iterdir():
        if p.is_file():file(str(p.relative_to(ROOT)),f'experiments/v11/stage1_20261104/tables/{p.name}',kind='derived_table_from_primary_JSON_or_saved_lengths',
            export_script='experiments/v11/stage1_20261104/tables/export_v11_tables.py',
            primary_source_sha256=digest(full.read_bytes()))
    file(str((WORK/'export_v11_tables.py').relative_to(ROOT)),'experiments/v11/stage1_20261104/tables/export_v11_tables.py')
    v12=[
      'v12_linux_prepare_20261106_release2/v12_linux_stage0_1_20261106.zip',
      'v12_stage0_2_prepare_20261106/v12_linux_stage0_2_20261106.zip',
      'v12_stage0_2_resource_repair_20261107/v12_stage0_2_resource_repair_20261107.zip',
      'v12_stage0_2_json_repair_20261107/v12_stage0_2_json_repair_20261107.zip',
      'v12_stage0_3_prepare_20261107/v12_linux_stage0_3_20261107.zip',
      'v12_stage1_2_20261107/release2/v12_linux_stage1_2_20261107.zip',
      'v12_stage2_continue_20261108/v12_stage2_continue_20261108.zip',
      'v12_stage3_20261108/v12_linux_stage3_20261108.zip',
      'v12_stage3_shared_20261108/v12_linux_stage3_shared_20261108.zip',
      'v12_stage3_shared50_20261108/v12_linux_stage3_shared50_20261108.zip',
      'v12_stage3_shared60_20261108/v12_linux_stage3_shared60_20261108.zip',
      'v12_stage3_switch_20261110/v12_linux_stage3_switch_20261110.zip']
    for source in v12:archive('outputs/'+source,'server_v12')
    archive('outputs/v12_linux_prepare_20261106/v12_linux_stage0_1_20261106.zip','server_v12','stage0_1_initial')
    archive('outputs/v12_stage1_2_20261107/v12_linux_stage1_2_20261107.zip','server_v12','stage1_2_initial')
    for source in ['v13_action_fix_stage0.zip','v13_stage0_report_repair_20261110.zip','v13_action_fix_stage1.zip','v13_stage1_pid_repair_20261110.zip','v13_stage1_shared60_20261110.zip']:
        archive('outputs/v13_action_fix_deployment/'+source,'server_v13')
    archive('outputs/v14_diag_deployment_20261112/v14_cpu_diag_20261112.zip','server_v14')
    parse_chat('6ff0cf9e-f6f1-499d-ab34-cd78222b8f18','server_v12','stage0_2')
    parse_chat('d6706d91-fae9-48b3-9875-411e1745aa92','server_v12','stage1B')
    parse_chat('3469b290-6bc4-48bf-b89f-3cdb09cb9952','server_v12','stage3','c7fbd6e515c4550a9971faea9ab3697a24393b97c93141f46d5e3d09b88a5c4c')
    parse_chat('d9245737-a5cf-4f23-b4a5-c80fba06bb4d','server_v13','stage1','70f32fe2b7e1d0e5d4f3949a3e62578d6804ca1473f5aa223992bbf3256cabf8')
    for name,top in [('v12_stage0_report_20261107','server_v12'),('v12_stage1_reports_20261108','server_v12'),('v12_stage2_report_20261108','server_v12')]:
        for p in (ROOT/'outputs'/name).glob('*.docx'):
            file(str(p.relative_to(ROOT)),f'{top}/editorial_reports/{p.name}',kind='exact_copy_of_previously_delivered_editorial_report',
                caveat='Assistant-prepared document based on received results; not an original server REPORT_ZH.md or completion receipt.')
    GAPS.extend([
      dict(stage='v12 stage0.1 / stage0.3',missing='original server reports, runtime config, completion receipts, and tables; only prior editorial summary is locally available'),
      dict(stage='v12 stage0.2',missing='original server completion receipt/config/result JSON; pasted report and determinism check available'),
      dict(stage='v12 stage1A / stage1B',missing='original runtime configs and completion receipts; stage1B report available only in terminal export; stage1A editorial summary available'),
      dict(stage='v12 stage2',missing='full original report, runtime config and receipts/result tables; prior editorial summary and upstream provenance available'),
      dict(stage='v12 stage3',missing='original runtime config, result JSON and original receipt file bytes; complete terminal report hash verified and exported receipt available'),
      dict(stage='v13 stage0',missing='original/repaired server report and completion receipts; solver/config/report repair code available'),
      dict(stage='v13 stage1',missing='server runtime RESULT/config/original completion file bytes; complete terminal report hash verified and exported receipt available'),
      dict(stage='v14',missing='no real diagnostic results supplied; code and 28-test synthetic acceptance only, intentionally not marked completed real diagnostics')])
    # Do not publish original absolute username-bearing paths in provenance.
    dump(WORK/'PREPARED_ENTRIES.json',ENTRIES)
    dump(WORK/'EXCLUDED_FILES.json',EXCLUDED)
    dump(WORK/'MISSING_SERVER_ARTIFACTS.json',GAPS)
    print(json.dumps(dict(files=len(ENTRIES),bytes=sum(e['snapshot_bytes'] for e in ENTRIES),
        transformed=sum(not e['exact_byte_copy'] for e in ENTRIES),excluded=len(EXCLUDED),
        sensitive_excluded=sum(e['reason']=='public_sensitive_information_detected' for e in EXCLUDED),
        roots={name:sum(e['snapshot'].startswith(name+'/') for e in ENTRIES) for name in ('experiments','server_v12','server_v13','server_v14')}),ensure_ascii=False))


if __name__=='__main__':main()
