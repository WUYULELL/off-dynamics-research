"""Read saved evaluation lengths only; stdlib, no experiment imports."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'outputs/v11_saall_report_20261104/stage1/STAGE1_RESULT.json'
GROUPS = ['SA-all', 'O1', 'T', 'raw1']
POINTS = [460000, 470000, 480000, 490000, 500000]
receipts = {}

def native(path):
    value = str(Path(path).resolve())
    return '\\\\?\\' + value if os.name == 'nt' and not value.startswith('\\\\?\\') else value

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    with open(native(path), 'rb') as stream:
        data = stream.read()
    receipts[str(Path(path).resolve())] = sha(data)
    return json.loads(data.decode('utf-8-sig'))

def write(name, data):
    with open(native(OUT / name), 'x', encoding='utf-8', newline='') as stream:
        stream.write(data)

def json_write(name, data):
    write(name, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def csv_write(name, rows):
    if not rows:
        return
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write(name, '\ufeff' + stream.getvalue())

def summarize(group, seed, point, lengths, expected):
    complete = len(lengths) == expected
    return dict(group=group, seed=seed, evaluation_point=point,
                status='已记录' if complete else '未记录',
                recorded_episodes=len(lengths), expected_episodes=expected,
                mean=statistics.mean(lengths) if complete else None,
                median=statistics.median(lengths) if complete else None,
                minimum=min(lengths) if complete else None,
                below_200_count=sum(x < 200 for x in lengths) if complete else None,
                below_200_fraction=sum(x < 200 for x in lengths)/expected if complete else None)

data = read(SOURCE)
records = [r for r in data['records'] if r['label'] in GROUPS]
assert len(records) == 40
assert {(r['label'], r['seed']) for r in records} == {(g, s) for g in GROUPS for s in range(201, 211)}
episodes, by_seed, by_point, missing = [], [], [], []
for group in GROUPS:
    for record in sorted((r for r in records if r['label'] == group), key=lambda r: r['seed']):
        seed = record['seed']
        path = Path(record['out']) / 'evaluations.json'
        events = read(path)['events']
        seed_lengths = []
        for point in POINTS:
            selected = [e for e in events if e.get('step') == point]
            assert len(selected) <= 1, (group, seed, point, 'duplicate checkpoint')
            values = selected[0].get('episode_lengths') if selected else None
            if values is None:
                values = []
                missing.append(dict(group=group, seed=seed, step=point, field='episode_lengths', status='未记录'))
            else:
                assert len(values) == 10
                assert all(type(v) is int and 0 < v <= 1000 for v in values)
                assert selected[0]['steps'] == sum(values)
                assert selected[0]['env_seeds'] == [500000 + 10*(seed-201) + i for i in range(10)]
            seed_lengths.extend(values)
            by_point.append(summarize(group, seed, point, values, 10))
            for index, value in enumerate(values):
                episodes.append(dict(group=group, seed=seed, evaluation_point=point,
                                     episode_index=index, environment_seed=500000+10*(seed-201)+index,
                                     episode_length=value))
        by_seed.append(summarize(group, seed, '460k–500k合并', seed_lengths, 50))

by_group = [summarize(g, '201–210合并', '460k–500k合并',
                     [e['episode_length'] for e in episodes if e['group'] == g], 500) for g in GROUPS]
by_group_point = [summarize(g, '201–210合并', p,
                     [e['episode_length'] for e in episodes if e['group'] == g and e['evaluation_point'] == p], 100)
                  for g in GROUPS for p in POINTS]
result = dict(read_only_CPU=True, new_training_updates=0, new_environment_steps=0,
              evaluation_points=POINTS, strict_short_episode_threshold=200,
              source_field='evaluations.json:events[].episode_lengths',
              aggregation='pool all saved episodes; 50 per seed, 500 per group; no independent-sample inference',
              groups=by_group, seeds=by_seed, seed_checkpoints=by_point,
              group_checkpoints=by_group_point, missing=missing)
json_write('EPISODE_LENGTH_RESULT.json', result)
for name, rows in [('by_group.csv', by_group), ('by_seed.csv', by_seed),
                   ('by_seed_checkpoint.csv', by_point), ('by_group_checkpoint.csv', by_group_point),
                   ('saved_episode_lengths.csv', episodes)]:
    csv_write(name, rows)

lines = ['# Hopper SA-all及对照：保存的评估回合长度只读诊断', '',
         '只读取已有逐回合评估记录，不训练、不构造环境、不重新评估。统计字段为 episode_lengths。',
         '评估点固定为460k、470k、480k、490k、500k；每点10回合。每种子合并50回合，每组10种子共500回合。',
         '均值、中位数和最小值的单位均为环境步；短回合比例严格按长度 < 200 计算。',
         '这是重复使用固定评估种子的描述统计，不将回合视为独立样本，也不改变已确认的H_act判定。', '',
         '## 按组汇总', '', '|组|记录回合数|均值|中位数|最小值|<200回合数|<200比例|',
         '|---|---:|---:|---:|---:|---:|---:|']

def table_row(row, include_seed=False):
    label = row['group'] + ('|' + str(row['seed']) if include_seed else '')
    if row['status'] == '未记录':
        return '|'+label+'|'+str(row['recorded_episodes'])+'|未记录|未记录|未记录|未记录|未记录|'
    return '|{}|{}|{:.3f}|{:g}|{}|{}|{:.2%}|'.format(
        label, row['recorded_episodes'], row['mean'], row['median'], row['minimum'],
        row['below_200_count'], row['below_200_fraction'])

lines.extend(table_row(r) for r in by_group)
lines.extend(['', '## 按种子列表', '', '|组|种子|记录回合数|均值|中位数|最小值|<200回合数|<200比例|',
              '|---|---:|---:|---:|---:|---:|---:|---:|'])
lines.extend(table_row(r, True) for r in by_seed)
lines.extend(['', '## 数据覆盖与出处', '',
              '缺失长度字段：' + ('无；40个种子记录、200个指定评估点全部记录长度，共2000回合。' if not missing else '见JSON中missing列表；相应统计标为“未记录”，未进行补评估。'),
              'SA-all使用正式报告接纳的201–206原完成记录、207重跑和208–210新记录；原中断207不纳入。',
              'by_seed_checkpoint.csv与by_group_checkpoint.csv另列每个评估点；saved_episode_lengths.csv列出原始长度及评估种子。',
              'INPUT_HASHES.json保存全部40份原始评估文件及出处清单的SHA256，读取后再次核对；不修改原文件。',
              '', '诊断完成后停止。SA-top正式训练与阶段2未启动，未上传GitHub。', ''])
write('REPORT_ZH.md', '\n'.join(lines))

for path, digest in receipts.items():
    with open(native(path), 'rb') as stream:
        assert sha(stream.read()) == digest, ('input changed', path)
json_write('INPUT_HASHES.json', receipts)
receipt = dict(passed=True, source_files=len(receipts), raw_evaluation_files=40,
               recorded_episodes=len(episodes), missing_fields=len(missing),
               inputs_unchanged=True, new_training_updates=0, new_environment_steps=0,
               experiment_modules_imported=False, timestamp=datetime.now(timezone.utc).isoformat(),
               outputs={p.name: sha(p.read_bytes()) for p in OUT.iterdir() if p.is_file()})
json_write('DIAGNOSTIC_COMPLETED.json', receipt)
print(json.dumps(dict(groups=by_group, missing=missing, input_files=len(receipts)), ensure_ascii=False))
