"""Controller helpers shared by the job kinds (adaptive-source batch)."""
import os
from pathlib import Path

from asrc_common import CODE, HOME, read_json, sha256_file, verify_frozen, verify_official, write_json, utc

OUT = CODE/'outputs'
LOCAL = os.environ.get('ASRC_LOCAL_TEST')        # path of a local-test json (synthetic data, CPU); never set on the server


def local_cfg():
    return read_json(LOCAL) if LOCAL else None


def status(cfg, phase, **kw):
    write_json(OUT/'STATUS.json', dict(time=utc(), job=cfg['job'], phase=phase, **kw))


def resolve_preflight(cfg):
    """Directory (server) of the first preflight job whose PREFLIGHT.json says all_passed."""
    if LOCAL:
        return Path(local_cfg()['preflight_dir'])
    for name in cfg['preflight_candidates']:
        d = HOME/'auto_jobs'/name/'code'
        f = d/'outputs'/'PREFLIGHT.json'
        if f.is_file() and read_json(f).get('all_passed') is True:
            return d
    raise AssertionError('no completed preflight job (all_passed) among %s' % cfg['preflight_candidates'])


def resolve_completed(candidates, what):
    """Directory of the first job (by candidate order) whose outputs/STATUS.json phase is 'completed'."""
    if LOCAL:
        return Path(local_cfg()['dirs'][what])
    for name in candidates:
        d = HOME/'auto_jobs'/name/'code'
        f = d/'outputs'/'STATUS.json'
        if f.is_file() and read_json(f).get('phase') == 'completed':
            return d
    raise AssertionError('%s: no completed job among %s' % (what, candidates))


def job_dir(name):
    if LOCAL:
        return Path(local_cfg()['dirs'][name])
    return HOME/'auto_jobs'/name/'code'


def preflight_files(pre):
    """COST_MANIFEST, SUPPORT and PREPARED manifests of the preflight job (server copies)."""
    cost = read_json(pre/'outputs'/'COST_MANIFEST.json')
    support = read_json(pre/'outputs'/'SUPPORT.json')
    prepared = read_json(pre/'outputs'/'PREPARED_MANIFEST.json')
    return cost, support, prepared


def cost_file_of(pre, cost_manifest, task_id):
    entry = cost_manifest['tasks'][task_id]
    path = pre/entry['cost_relpath']
    assert path.is_file() and sha256_file(path) == entry['cost_sha256'], 'cost file of %s missing or changed' % task_id
    return path, entry


def check_common(cfg):
    files = verify_frozen()
    official = verify_official()
    return dict(frozen_files=len(files), official=official)
