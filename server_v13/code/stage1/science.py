"""Read-only data routing and reuse of frozen definitions."""
import ast
import hashlib
from pathlib import Path
import random
import sys

from common import read, sha


HERE = Path(__file__).resolve().parent
V9 = 'source_snapshot/oracle_diag/v9_hopper_label_headroom/stage3'
V9_SELECTIONS = 'records/outputs/v9_hopper_label_headroom_20261031/stage3/selections.json'
TARGET_KEYS = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = {entry['path']: entry for entry in read(self.root / 'MANIFEST.json')['files']}
        self.tasks = read(self.path('cluster/paths.example.json'))['tasks']

    def path(self, relative):
        path = (self.root / relative).resolve()
        entry = self.manifest[relative]
        assert self.root in path.parents and path.stat().st_size == entry['bytes']
        assert sha(path) == entry['sha256'], relative
        return path

    def spec(self, task):
        values = dict(self.tasks[task])
        path = self.path(values['xml'])
        values.update(xml=str(path), xml_sha256=self.manifest[self.tasks[task]['xml']]['sha256'])
        return values


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names) and {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def transitions(path, source=False, actor_only=False):
    import h5py
    import numpy as np

    with h5py.File(path, 'r') as data:
        n = len(data['observations'])
        if source:
            # D4RL qlearning_dataset's v2 row selection with terminate_on_end=False.
            timeouts = np.asarray(data['timeouts']).reshape(-1)
            assert len(timeouts) == n and np.isin(timeouts, [0, 1]).all()
            rows = np.flatnonzero(~timeouts[:-1].astype(bool))
        else:
            rows = np.arange(n, dtype=np.int64)
        keys = TARGET_KEYS[:2] if actor_only else TARGET_KEYS
        arrays = {key: np.asarray(data[key][()], dtype=np.float32)[rows] for key in keys}
    assert len(rows) and all(np.isfinite(array).all() for array in arrays.values())
    if not actor_only:
        assert np.isin(arrays['terminals'], [0, 1]).all()
        assert arrays['observations'].shape == arrays['next_observations'].shape
    return arrays, rows


def validate_dimensions(arrays, spec):
    n = len(arrays['observations'])
    assert arrays['observations'].shape == (n, spec['state_dim'])
    assert arrays['actions'].shape == (n, spec['action_dim'])


def select_trajectories(project):
    import h5py
    import numpy as np

    reference = read(project.path(V9_SELECTIONS))
    result = {}
    for task, record in reference.items():
        all_segments = sorted(record['segments'], key=lambda row: row['segment'])
        complete = [row for row in all_segments if row['complete_boundary']]
        assert len(complete) == 7 and len(all_segments) == 8 and not all_segments[-1]['complete_boundary']
        assert [row['segment'] for row in complete] == list(range(1, 8))
        with h5py.File(project.path(project.tasks[task]['target_dataset']), 'r') as data:
            rewards = np.asarray(data['rewards']).reshape(-1)
        assert len(rewards) == 5000 and all_segments[-1]['end_row'] == 4999
        next_row = 0
        for row in all_segments:
            start, end = row['start_row'], row['end_row'] + 1
            assert start == next_row and end - start == row['length']
            assert np.isclose(rewards[start:end].sum(dtype=np.float64), row['raw_return'], rtol=0, atol=1e-8)
            next_row = end
        order = np.random.RandomState(777).permutation(len(complete)).tolist()
        ordered = [complete[index] for index in order]
        subsets = {}
        for n in (2, 4, 7):
            selected = ordered[:n]
            # Order the transitions by original row, as in the original BC sampler.
            rows = sorted(index for row in selected for index in range(row['start_row'], row['end_row'] + 1))
            assert len(set(rows)) == len(rows) and max(rows) < all_segments[-1]['start_row']
            subsets[str(n)] = dict(
                trajectory_ids=[row['segment'] for row in selected], rows=rows, n_rows=len(rows),
                mean_raw_trajectory_return=float(np.mean([row['raw_return'] for row in selected])),
                individual_raw_returns=[row['raw_return'] for row in selected],
                mean_normalized_trajectory_return=float(np.mean([row['normalized_return'] for row in selected])),
            )
        assert set(subsets['2']['rows']) < set(subsets['4']['rows']) < set(subsets['7']['rows'])
        result[task] = dict(permutation=[row['segment'] for row in ordered], permutation_seed=777,
                            excluded_tail=all_segments[-1], subsets=subsets)
    assert set(result) == {'hopper-morph', 'walker2d-morph'}
    return result


def bc_definitions(project, rows):
    import numpy as np
    import torch

    policy_path = project.path('source_snapshot/algo/offline_offline/iql_unc.py')
    sys.path.insert(0, str(policy_path.parents[1]))
    from offline_offline import iql_unc
    assert Path(iql_unc.__file__).resolve() == policy_path
    namespace = dict(__name__='v12_bc', np=np, torch=torch, random=random, hashlib=hashlib,
                     Policy=iql_unc.Policy, task_spec=project.spec, ids=lambda job: rows, sha=sha)
    extract(project.path(V9 + '/runtime.py'), {'seed', 'env'}, namespace)
    extract(project.path(V9 + '/learner.py'), {'Actor', 'Sampler'}, namespace)
    extract(project.path(V9 + '/train.py'), {'evaluate'}, namespace)
    return namespace


def rows_digest(rows):
    import numpy as np
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()
