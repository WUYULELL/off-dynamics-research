"""Export small V11 tables without loading a model or running an environment.

Scores come directly from STAGE1_RESULT.json. That primary JSON records no
episode lengths. Lengths, where available, come from the separately accepted
EPISODE_LENGTH_RESULT.json; missing fields are never inferred from scores.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_csv(path, fields, rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--lengths',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result_hash,length_hash=sha(a.result),sha(a.lengths)
    x=json.loads(a.result.read_text(encoding='utf-8-sig'))
    lengths=json.loads(a.lengths.read_text(encoding='utf-8-sig'))
    assert x['passed'] and x['raw_records']==70 and not x['stage2_started']
    assert lengths['new_training_updates']==lengths['new_environment_steps']==0
    a.output.mkdir(parents=True,exist_ok=False)
    primary=[]
    for r in x['records']:
        primary.append(dict(group=r['label'],seed=r['seed'],Last5=r['Last5'],Final=r['Final'],
            length_status='not recorded in primary JSON',length_mean='',length_median='',length_minimum='',length_fraction_under_200=''))
    primary.sort(key=lambda r:(r['group'],r['seed']))
    fields=list(primary[0])
    output_csv(a.output/'scores_direct_from_primary.csv',fields,primary)
    by_seed={(r['group'],r['seed']):r for r in lengths['seeds']}
    final_seed={(r['group'],r['seed']):r for r in lengths['seed_checkpoints'] if r['evaluation_point']==500000}
    supplemental=[]
    for row in primary:
        r={k:row[k] for k in ('group','seed','Last5','Final')}
        recorded=False
        for window,lookup in [('Last5',by_seed),('Final',final_seed)]:
            s=lookup.get((row['group'],row['seed']))
            r[window+'_length_status']='saved episodes, separate diagnostic JSON' if s and s['status']=='已记录' else 'not recorded in supplied length diagnostic'
            for dst,key in [('mean','mean'),('median','median'),('minimum','minimum'),('fraction_under_200','below_200_fraction')]:
                r[window+'_length_'+dst]=s[key] if s and s['status']=='已记录' else ''
            recorded=recorded or s is not None
        r.update(score_source_sha256=result_hash,length_source_sha256=length_hash if recorded else '')
        supplemental.append(r)
    output_csv(a.output/'per_seed_scores_and_lengths.csv',list(supplemental[0]),supplemental)
    output_csv(a.output/'SA_all_per_seed.csv',list(supplemental[0]),[r for r in supplemental if r['group']=='SA-all'])
    group_rows=[]
    length_groups={r['group']:r for r in lengths['groups']}
    final_groups={r['group']:r for r in lengths['group_checkpoints'] if r['evaluation_point']==500000}
    for window,record in x['statistics'].items():
        for name,values in record['groups'].items():
            row=dict(window=window,group=name,**values)
            lens=(length_groups if window=='Last5' else final_groups).get(name)
            row['length_status']='saved episodes, separate diagnostic JSON' if lens else 'not recorded in supplied length diagnostic'
            for dst,key in [('mean','mean'),('median','median'),('minimum','minimum'),('fraction_under_200','below_200_fraction')]:
                row['length_'+dst]=lens[key] if lens else ''
            group_rows.append(row)
    keys=[]
    for row in group_rows:
        for k in row:
            if k not in keys:keys.append(k)
    # Group objects contain only scalar summary statistics. Nested fields are
    # serialized explicitly rather than silently flattened or discarded.
    for row in group_rows:
        for k,v in list(row.items()):
            if isinstance(v,(dict,list)):row[k]=json.dumps(v,ensure_ascii=False,sort_keys=True)
    output_csv(a.output/'group_statistics.csv',keys,group_rows)
    output_csv(a.output/'saved_length_by_seed.csv',list(lengths['seeds'][0]),lengths['seeds'])
    output_csv(a.output/'saved_length_by_group.csv',list(lengths['groups'][0]),lengths['groups'])
    final_rows=[r for r in lengths['seed_checkpoints'] if r['evaluation_point']==500000]
    output_csv(a.output/'saved_Final_length_by_seed.csv',list(final_rows[0]),final_rows)
    meta=dict(primary_original_bytes=a.result.stat().st_size,primary_sha256=result_hash,
        episode_length_source_sha256=length_hash,export_script_sha256=sha(__file__),
        primary_score_rows=len(primary),SA_all_rows=10,primary_has_episode_lengths=False,
        length_provenance='Separate accepted read-only diagnostic; Last5 460k-500k pooled recorded episodes. Not derived or inferred from primary scores.',
        absent_groups='Groups not covered by the saved length diagnostic are explicitly missing; no reevaluation.',
        original_files_unchanged=sha(a.result)==result_hash and sha(a.lengths)==length_hash,
        new_training_updates=0,new_environment_steps=0,
        outputs={f.name:sha(f) for f in a.output.glob('*.csv')})
    (a.output/'EXPORT_PROVENANCE.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=meta['original_files_unchanged'],score_rows=len(primary),files=len(meta['outputs']),training_updates=0,environment_steps=0)))


if __name__=='__main__':main()
