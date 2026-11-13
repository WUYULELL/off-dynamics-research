"""Build an additive, small-file archive from a user-provided server export.

Standard library only. Never imports experiment code, connects to a server,
loads a model, creates a simulator, or changes the input ZIP/source outputs.
Usage: python build_supplement.py --zip export.zip --output new_payload
"""
import argparse
import collections
import csv
import hashlib
import io
import ipaddress
import json
import posixpath
from pathlib import Path, PurePosixPath
import re
import statistics
import zipfile

LIMIT = 5_000_000
TAG = 'results_20261113'
AUDIT = 'experiments/publication_supplement_20261113'
ZIP_SHA = '95c5385da03d751cfa7d743190fbf1ae34f22abe10330bc21ec3becc528f813a'
ROOT_MAP = {'v12_linux':'server_v12', 'v13_action_fix':'server_v13', 'v14_diag':'server_v14'}

def sha(data): return hashlib.sha256(data).hexdigest()
def jbytes(obj): return (json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
def csvbytes(rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n')
    w.writeheader();w.writerows(rows);return f.getvalue().encode('utf-8')

def private_types(text):
    rules={
      'server_login_identifier':r'\becs-instance-[\w-]+\b|\bcs_dsns_(?!public\b)[A-Za-z0-9_]+\b',
      'unredacted_windows_user':r'''[A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)(?!<USER>)[^\\/\s"<>]+''',
      'unredacted_server_user':r'''(?:/mnt/cs_dsns_public/|/home/)(?!<USER>)[A-Za-z][A-Za-z0-9_.-]+(?=/|["'\s]|$)''',
      'private_key':r'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|ssh-(?:rsa|ed25519)\s+[A-Za-z0-9+/]{30,}',
      'token':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,})\b',
      'credential_literal':r'''(?i)(?:password|passwd|access_token|api_key)\s*["']?\s*[:=]\s*["'][^"']{5,}["']''',
      'other_process_data':r'''(?is)["'](?:other_pids|visible_pids|other_GPU_pids|external_GPU_pids_at_release)["']\s*:\s*\[\s*\d|GPU-[a-fA-F0-9-]{20,}\s*,\s*\d{3,}\s*,''',
    }
    found=[k for k,p in rules.items() if re.search(p,text)]
    for line in text.splitlines():
        for m in re.finditer(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',line):
            try: ipaddress.IPv4Address(m.group())
            except ValueError: continue
            # Version strings in requirements and the pinned mujoco-py version.
            if m.group()=='2.1.2.14' or re.fullmatch(r'\s*[A-Za-z0-9_.\[\],-]+==[A-Za-z0-9_.+!-]+\s*',line):continue
            found.append('IPv4_address');break
    return sorted(set(found))

def report_tables(text):
    tables=[];block=[];section=''
    def flush():
        if len(block)<3:return
        split=lambda s:[x.strip() for x in s.strip().strip('|').split('|')]
        header=split(block[0])
        if not all(re.fullmatch(r':?-+:?',x.replace(' ','')) for x in split(block[1])):return
        rows=[split(x) for x in block[2:]]
        assert all(len(x)==len(header) for x in rows)
        tables.append({'section':section,'header':header,'rows':rows})
    for line in text.splitlines()+['']:
        if line.startswith('|'):block.append(line)
        else:
            flush();block=[]
            if line.startswith('#'):section=line.lstrip('#').strip()
    return tables

def lengths(values):
    if not values:return dict(mean='未记录',median='未记录',minimum='未记录',fraction_under_200='未记录')
    return dict(mean=statistics.mean(values),median=statistics.median(values),minimum=min(values),fraction_under_200=sum(x<200 for x in values)/len(values))

def report_seed_rows(tables):
    output=[]
    for t in tables:
        h=t['header']
        if '种子' not in h or 'Last5' not in h or 'Final' not in h:continue
        for r in t['rows']:
            d=dict(zip(h,r));seed=d['种子']
            if not seed.isdigit():continue
            row={'task':d.get('任务','walker2d-morph'),'group':d.get('组',d.get('数据','')),'seed':int(seed),'Last5':d['Last5'],'Final':d['Final']}
            for window in ('Last5','Final'):
                k=next((k for k in h if k.startswith(window+' 长度')),None)
                if k:
                    v=[x.strip() for x in d[k].split('/')];assert len(v)==4
                    row.update({window+'_length_mean':v[0],window+'_length_median':v[1],window+'_length_minimum':v[2],window+'_length_fraction_under_200':float(v[3].rstrip('%'))/100})
            if '物理 GPU' in d:row['physical_GPU_index']=d['物理 GPU']
            output.append(row)
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--output',required=True);p.add_argument('--collector',required=True)
    args=p.parse_args();archive=Path(args.zip);out=Path(args.output)
    assert not out.exists(),'Use a new output directory'
    assert sha(archive.read_bytes())==ZIP_SHA
    out.mkdir(parents=True)
    entries=[];derived=[];checks=[];stage_counts=collections.Counter()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('EXPORT_MANIFEST.json'))
        meta={x['snapshot']:x for x in manifest['files']}
        assert len(z.namelist())==len(set(z.namelist()))
        assert set(z.namelist())==set(meta)|{'EXPORT_MANIFEST.json'}
        for n in z.namelist():
            q=PurePosixPath(n)
            assert not q.is_absolute() and '..' not in q.parts and ':' not in n and '\\' not in n
            assert ((z.getinfo(n).external_attr>>16)&0o170000)!=0o120000
        excluded=[dict(x,publication_decision='未上传；原件保留在服务器；内容不在收到的 ZIP 内',review_scope='只收到排除元数据，未复查被排除文件正文') for x in manifest['excluded']]
        def target(n):
            q=PurePosixPath(n);assert q.parts[1]=='outputs'
            return str(PurePosixPath(ROOT_MAP[q.parts[0]])/TAG/PurePosixPath(*q.parts[2:]))
        def put(path,data,provenance=None):
            assert len(data)<=LIMIT
            assert not private_types(data.decode('utf-8')),path
            dest=out/path;dest.parent.mkdir(parents=True,exist_ok=True)
            assert not dest.exists(),path;dest.write_bytes(data)
            entry=dict(snapshot=path,snapshot_bytes=len(data),snapshot_sha256=sha(data))
            if provenance:entry.update(provenance)
            entries.append(entry)
        for n,x in sorted(meta.items()):
            data=z.read(n);assert sha(data)==x['snapshot_sha256'] and len(data)<=LIMIT
            bad=private_types(data.decode('utf-8'))
            reason=None
            if bad:reason='敏感信息；整文件排除'
            elif re.fullmatch(r'progress_\d+\.json',PurePosixPath(n).name):reason='运行中进度日志；不属于报告、完成凭据或结果表'
            if reason:
                excluded.append(dict(source=x['source'],bytes=x['bytes'],source_sha256=x['source_sha256'],export_sha256=x['snapshot_sha256'],reason=reason,finding_types=bad,publication_decision='未上传；收到的 ZIP 和服务器原件均保持不变'))
                continue
            put(target(n),data,dict(source=x['source'],source_bytes=x['bytes'],source_sha256=x['source_sha256'],source_archive=archive.name,source_archive_sha256=ZIP_SHA,source_member=n,exported_member_sha256=x['snapshot_sha256'],transformations=x['transformations'],exact_byte_copy_of_original=x['exact_byte_copy'],exact_byte_copy_of_export=True,kind='server_export_copy'))
            stage_counts['/'.join(PurePosixPath(n).parts[:3])]+=1
        # Check receipt hashes against original hashes, not redacted public bytes.
        all_meta={**{k:v for k,v in meta.items()},**{x['source'].removeprefix('${HOME}/'):x for x in manifest['excluded']}}
        for n,x in meta.items():
            if not n.endswith('.json'):continue
            obj=json.loads(z.read(n))
            if not isinstance(obj,dict):continue
            links=[]
            for k in ('evidence_sha256','artifacts_sha256'):
                if isinstance(obj.get(k),dict):links.extend(obj[k].items())
            for k,basename in [('completed_sha256','COMPLETED.json'),('report_sha256','STAGE3_REPORT_ZH.md' if '/stage3_' in n else 'REPORT_ZH.md'),('result_sha256','STAGE3_RESULT.json' if '/stage3_' in n else 'RESULT.json'),('config_sha256','config.json')]:
                if k in obj:
                    if PurePosixPath(n).name=='STAGE0_1_APPROVAL.json' and k=='report_sha256':
                        basename='../stage0_1_20261106/REPORT_ZH.md'
                    links.append((basename,obj[k]))
            for rel,h in links:
                logical=posixpath.normpath(str(PurePosixPath(n).parent/rel))
                if logical in all_meta:
                    ok=h==all_meta[logical]['source_sha256'];assert ok,(n,logical)
                    checks.append(dict(receipt=target(n),referenced_source='${HOME}/'+logical,matches_original_sha256=ok,referenced_original_sha256=h))
        reports={}
        for n in sorted(meta):
            if PurePosixPath(n).name not in ('REPORT_ZH.md','STAGE3_REPORT_ZH.md'):continue
            if not (out/target(n)).exists():continue
            tables=report_tables(z.read(n).decode('utf-8'));base=str(PurePosixPath(target(n)).parent/'tables')
            rec=[]
            for i,t in enumerate(tables,1):
                rows=[dict(zip(t['header'],r)) for r in t['rows']]
                path=base+f'/report_table_{i:02d}.csv'
                if rows:put(path,csvbytes(rows),dict(kind='derived_report_table',source=meta[n]['source'],source_sha256=meta[n]['source_sha256'],source_public_sha256=meta[n]['snapshot_sha256'],precision='原报告显示精度；未从被排除的 RESULT JSON 导出',export_script=AUDIT+'/tools/build_supplement.py'))
                rec.append(dict(table=path,section=t['section'],rows=len(rows)))
            seeds=report_seed_rows(tables)
            if seeds:
                expected=5 if '/stage1A_' in n else 30 if '/stage1B_' in n else 15 if '/stage3_switch_' in n else None
                if expected:assert len(seeds)==expected
                path=base+'/per_seed_scores_and_lengths_report_precision.csv'
                put(path,csvbytes(seeds),dict(kind='derived_report_seed_table',source=meta[n]['source'],source_sha256=meta[n]['source_sha256'],source_public_sha256=meta[n]['snapshot_sha256'],precision='原报告显示精度；百分比除以100转比例',rows=len(seeds),export_script=AUDIT+'/tools/build_supplement.py'))
            put(base+'/TABLE_SOURCES.json',jbytes(dict(source_report=target(n),source_original_sha256=meta[n]['source_sha256'],source_public_sha256=meta[n]['snapshot_sha256'],export_script=AUDIT+'/tools/build_supplement.py',export_script_sha256=sha(Path(__file__).read_bytes()),tables=rec,per_seed_rows=len(seeds),note='逐表保留原报告精度，不填造未记录字段。')),dict(kind='generated_derivation_receipt'))
            reports[n]={'tables':len(tables),'seed_rows':len(seeds)}
        # V13 full precision exports directly from the actual RESULT JSON.
        n='v13_action_fix/outputs/stage1_20261110_shared60/RESULT.json'
        result=json.loads(z.read(n));assert len(result['runs'])==25 and len(result['groups'])==5
        rows=[]
        for job,r in sorted(result['runs'].items()):
            row=dict(job=job,group=r['variant'],seed=r['seed'],Last5=r['last5'],Final=r['final'])
            for window,key in [('Last5','last5_episode_lengths'),('Final','final_episode_lengths')]:
                row.update({window+'_length_'+k:v for k,v in lengths(r.get(key)).items()})
            rows.append(row)
        base=str(PurePosixPath(target(n)).parent/'tables')
        provenance=dict(kind='derived_JSON_table',source=meta[n]['source'],source_sha256=meta[n]['source_sha256'],source_public_sha256=meta[n]['snapshot_sha256'],export_script=AUDIT+'/tools/build_supplement.py',export_script_sha256=sha(Path(__file__).read_bytes()),precision='JSON 原数值；回合长度从该 JSON 内数组直接统计')
        put(base+'/per_seed_scores_and_lengths_full_precision.csv',csvbytes(rows),provenance)
        groups=[]
        for group,r in sorted(result['groups'].items()):
            row=dict(group=group,worst_Last5_seed=r['last5_worst_seed'])
            for window,key in [('Last5','last5'),('Final','final')]:
                row.update({window+'_'+k:v for k,v in r[key].items()})
                row.update({window+'_length_'+k:v for k,v in r[key+'_lengths'].items()})
            groups.append(row)
        put(base+'/group_statistics_full_precision.csv',csvbytes(groups),provenance)
        put(AUDIT+'/RECEIVED_EXPORT_MANIFEST.json',z.read('EXPORT_MANIFEST.json'),dict(kind='exact_export_manifest',source=archive.name+'!/EXPORT_MANIFEST.json',source_sha256=sha(z.read('EXPORT_MANIFEST.json'))))
        put(AUDIT+'/EXCLUDED_FILES.json',jbytes(excluded),dict(kind='generated_exclusion_list'))
        put(AUDIT+'/RECEIPT_HASH_LINKS_VERIFIED.json',jbytes(dict(passed=True,checked=len(checks),scope='收到的小文件及排除元数据之间的原始哈希交叉核验；未加载模型或重跑科学审计',checks=checks)),dict(kind='generated_receipt_verification'))
        code=Path(__file__).read_bytes();put(AUDIT+'/tools/build_supplement.py',code,dict(kind='publication_only_script',source='generated:build_supplement.py',source_sha256=sha(code)))
        c=Path(args.collector).read_bytes();put(AUDIT+'/tools/collect_missing_small_files.py',c,dict(kind='exact_export_script',source='${PROJECT_ROOT}/outputs/github_missing_collect_20261112/collect_missing_small_files.py',source_sha256=sha(c),exact_byte_copy_of_original=True))
        summaries={
          'server_v12':'''# V12 服务器原始结果补充（2026-11-13）

本目录补齐上一次归档中尚未收到的服务器原始报告、实际配置、完成凭据与逐运行审计凭据。自有部署/运行代码已在上一提交 c346377 的 server_v12/code 和 historical_code 归档；本次不重复复制或修改。ROOT 来源为 https://github.com/cristianhtvc/ROOT ，固定提交 0ae9c05d70ed369c16732c71abda59b54b559da9；改动说明见 ../ROOT_REPRODUCTION_NOTES.md。

- 阶段 0.1：两任务各有放回回放 10,000 条；0.2：2×10k 和一次20k短跑，同机确定性核对通过；0.3：吞吐测量。它们是迁移验收，不是正式训练结果。历史 4/6 并发配置原样保留，不代表当前使用许可。
- 阶段 1A：5项源 BC，Last5 13.3235，短于200步100%。阶段1B：30项；Hopper 2/4/7轨迹22.9670/36.8060/51.0555，Walker 24.9557/36.4685/36.6522。
- 阶段2：10k VAE+20k ROOT迁移短跑，两点评估14.4609/14.3542；不是100万步正式结论。
- 阶段3：ROOT/T/BC-all各5个种子，15项已审计；Last5 32.7512/41.9092/37.6788。预设 H_root 不成立。8项复用+7项新执行；原中断尝试未算正式结果，停止等待用户决定。

阶段3按已确认的上游生效行为运行：Q权重广播不修复，actor不传权重，proportion过滤注释；源相邻观测与阶段2显式next在513条终止转移上不同，目标显式next保留5000行、与上游有7条差异。真实批次只读Q损失为23.8276978 / 22.1358051 / 23.8468361。权重只算一次并复用，分布对照和哈希见原报告及凭据。

含地址或他人进程信息的完整 RESULT/资源凭据未上传，见 ../../experiments/publication_supplement_20261113/EXCLUDED_FILES.json。阶段1A、1B、3的逐种子表直接从已验证原报告导出，只保留报告显示精度，不冒充来自被排除JSON的全精度表。原始配置和报告仅按导出清单替换路径用户名；不改科学数值。
''',
          'server_v13':'''# V13 服务器原始结果补充（2026-11-13）

本目录补齐阶段0原故障、修复后的报告和完成凭据，以及阶段1实际配置、全量缓存统计、独立回放验收、25项BC审计和正式阶段报告。求解器、缓存/回放、BC入口、队列及修复代码已在上一提交 c346377 的 server_v13/code 归档。

阶段0原报告在最近邻距离表访问缺少的min字段时退出；修复仅从已有数组补算分档最小值并核对既有产物，新增仿真/训练0。该事实记录在修复报告中，原失败凭据保留。

阶段1固定 e1<=0.0840，依据阶段0残差而非策略回报。999,998行中保留503,298行（50.3299006598%），独立10,000条回放验收。5组×601–605、各200k，25项通过审计（1项旧首跑+24项接续）。Last5：BC-fix13.5451、BC-mix13.5741、BC-all52.1258、BC-fix-sel12.6901、BC-mix-sel14.4382。

BC-fix低于20且短回合100%，触发阶段1预设低表现条件；不是尚未执行的阶段2正式H_fix检验。源动作原始触边0%，修正触边58.4452%；全量e1均值0.1213205574。边界/收敛分组、动作改变量、终止比例、保留比例见原RESULT及缓存统计。

tables/per_seed_scores_and_lengths_full_precision.csv 和 group_statistics_full_precision.csv 直接从本目录原始RESULT JSON导出，含来源与脚本SHA-256。报告表的副本另标report_precision，不混用精度。阶段2未启动，停止等待用户决定。
''',
          'server_v14':'''# V14 服务器真实诊断结果补充（2026-11-13）

本目录提供服务器实际A/C诊断、B0逐轨迹结果、配置和完成凭据，补充上一提交中只有代码与28项合成自检的状态。原代码和自检见 ../code，不执行新诊断。

真实停点：A、C完成，B0不通过，B1/B2未启动。完成凭据的completed表示已按授权停下并产出报告，all_three_diagnostics_completed=false，不表示全部诊断通过。新增训练0、优化器0、GPU作业0；原CPU诊断含200回合评估及83,723环境步，本次归档新增环境步0。

A：目标MSE均值BC-all0.0001155376、BC-fix0.7255106、BC-mix0.0007660779、BC-mix-sel0.0006942308；mix/all=6.630551，达到预设相对MSE分支。混合项64+64采样和相加日志核查通过，不能据此断言目标项被遗漏。轨迹与动作比较仅作描述。

B0：8条记录段都到达末尾、总长度5000，但每条都有原始下一观测最大绝对误差超过固定2e-6；最大约4.02038e-5，因此按用户门槛停止B1/B2，未放宽阈值。不能从未运行的B1/B2推断源轨迹不可跟踪。

C：999,998行e1均值约0.12132055；分步位置统计仅描述，不能单凭位置解释步态因果。详细数值以REPORT_ZH、A/B/C子目录原JSON和CSV为准。含地址的根RESULT未收到且不上传，子诊断RESULT可用。停止等待用户审阅，未启动V13阶段2。
'''
        }
        for root,text in summaries.items():put(root+'/'+TAG+'/README.md',text.encode('utf-8'),dict(kind='generated_archive_summary'))
        status=dict(previous_commit='c346377af52b91191976da8b07c35616116e494d',archive_sha256=ZIP_SHA,source_files_verified=661,export_exclusions=10,additional_sensitive_exclusions=3,progress_log_exclusions=20,server_files_uploaded=sum(stage_counts.values()),received_stage_directories=dict(stage_counts),previous_gap_list_unchanged=True,code_already_published=True,remaining_intentional_omissions='含敏感信息原件按清单保留；模型/数据/缓存/逐步日志不在上传范围；V14 B1/B2按B0门槛未运行，无对应结果。',formal_conclusions=dict(V12_H_root='不成立',V13_stage2_H_fix='未检验',V14_B0='不通过，A/C完成，B1/B2未运行'),training_updates_started=0,environment_steps_started=0,server_connection_attempts=0,original_outputs_modified=False,report_exports=reports)
        assert sum(stage_counts.values())==638
        put(AUDIT+'/SUPPLEMENT_STATUS.json',jbytes(status),dict(kind='generated_archive_status'))
        notice='''# GitHub 研究材料补充交付（2026-11-13）

本次追加 server_v12/results_20261113/、server_v13/results_20261113/、server_v14/results_20261113/ 与本目录。上次提交已上传Windows V11、种子207与回合长度诊断，以及V12–V14部署代码。本次收到服务器原件后补足报告、配置、完成/审计凭据、小表；旧文件与旧缺口清单保持原样，以SUPPLEMENT_STATUS.json说明当前状态。

收到ZIP的SHA-256为95c5385da03d751cfa7d743190fbf1ae34f22abe10330bc21ec3becc528f813a，661个成员全部核对。公开638个原件的导出副本。10个服务器端已排除文件未收到正文；另外3个文件包含他人的GPU PID，整文件排除；20个progress文件是运行中进度日志，不上传。排除清单记录原路径模板、原字节数和原SHA，不暴露敏感值，也不删除原件。

已替换用户名的导出字节不同于服务器原始字节。SNAPSHOT_SOURCES.json逐项记录source_sha256（服务器原件）、exported_member_sha256与snapshot_sha256（公开副本）、替换记录和ZIP来源。完成凭据内引用的哈希维持原样，按source_sha256交叉核验；不谎称公开副本哈希与原件相同。

结果摘要和停点分别见三个results_20261113/README.md及原报告。V12阶段3 H_root不成立；V13阶段1是BC诊断，阶段2未运行；V14 A/C完成、B0失败、B1/B2未运行。现有V11 H_act在Hopper不成立。全部训练范围与限制不因归档改变。

未上传的完整JSON不用于假装生成全精度表。V12小表从原报告直接导出，标明报告精度；V13阶段1另从收到的RESULT JSON直接生成全精度25种子和5组表。工具、来源和脚本哈希均保留。不上传压缩包本体、模型、数据、缓存数组、逐步损失或已安装环境。本次没有实验、服务器连接、重新评估或环境安装。
'''
        put(AUDIT+'/README.md',notice.encode('utf-8'),dict(kind='generated_delivery_note'))
        # Snapshot manifests describe content, avoiding impossible self-hashes.
        for root in ROOT_MAP.values():
            prefix=root+'/'+TAG+'/'
            selected=[e for e in entries if e['snapshot'].startswith(prefix)]
            put(prefix+'SNAPSHOT_SOURCES.json',jbytes(dict(files=selected,source_archive_sha256=ZIP_SHA,source_path_template='${HOME} denotes original server user home; absolute path usernames are <USER>',note='Original receipt hashes refer to original server bytes; source and public bytes are separately recorded.')),dict(kind='generated_source_manifest'))
        put(AUDIT+'/SNAPSHOT_SOURCES.json',jbytes(dict(files=entries.copy(),excluded=excluded,archive_sha256=ZIP_SHA,manifest_self_hash_policy='This manifest excludes its own hash; final Git tree and local review manifest bind its bytes.')),dict(kind='generated_master_manifest'))
    # All output files remain reviewable locally before publication.
    final=[dict(path=e['snapshot'],bytes=e['snapshot_bytes'],sha256=e['snapshot_sha256']) for e in entries]
    (out.parent/'FINAL_PAYLOAD_MANIFEST.json').write_bytes(jbytes(final))
    (out.parent/'ADDED_FILES.txt').write_text('\n'.join(e['path'] for e in final)+'\n',encoding='utf-8')
    (out.parent/'PREPUBLICATION_CHECK.json').write_bytes(jbytes(dict(passed=True,files=len(final),bytes=sum(e['bytes'] for e in final),largest_file_bytes=max(e['bytes'] for e in final),source_files=638,excluded=len(excluded),receipt_hash_links_checked=len(checks),source_zip_unchanged=sha(archive.read_bytes())==ZIP_SHA,experiments_started=False)))
    with zipfile.ZipFile(out.parent/'PUBLIC_REVIEW_BUNDLE_20261113.zip','x',compression=zipfile.ZIP_DEFLATED) as review:
        for e in final:review.write(out/e['path'],e['path'])
    print(json.dumps(dict(prepared_files=len(final),server_copies=638,exclusions=len(excluded),receipt_hash_links_checked=len(checks),derived_reports=len(reports)),ensure_ascii=True))

if __name__=='__main__':main()
