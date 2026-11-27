"""Shared helpers for the adaptive-source batch (j013-j016).

Python 3.8 compatible; only the standard library at import time. Outputs never contain absolute paths:
paths are written relative to code/ or as ~/... (see rel()).
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

CODE = Path(__file__).resolve().parent
HOME = Path.home()
SEED_PREFIX = 'ASRC-20261127'

# Official OTDF files (commit 41e2e626844bd6b65e2316a418ce4d6df4bbe775), SHA-256 of the bytes in the official
# GitHub tree (LF line endings). The copies under reproductions/OTDF in this repository are a Windows checkout of
# the same files (CRLF); with CR removed they are byte-identical to these.
OTDF_COMMIT = '41e2e626844bd6b65e2316a418ce4d6df4bbe775'
OTDF_OFFICIAL_SHA256 = {
    'otdf_official/algo/OTDF.py': 'af78c07d83abc5ca41e8e80581e788d91fa6d46f53b1078920dc5f4f88fcc898',
    'otdf_official/algo/utils.py': '3eb5cd0249b91216ad507ac44c24a3eca6ce914e8c11334c3faa3efe7cc64801',
    'otdf_official/info.py': '487707945b042a7cd93f9a0d0802f516fe7975309d2f3319a3e785652a4a47a3',
    'otdf_official/make_otdf_cost.py': '8a7dec164f9df030440f9d4024fb64d469906b05533070324027c75093bfec37',
    'otdf_official/train_otdf.py': '203347335831b10495037862148eabe6b1d0af15489990bf21a8a7e761501f88',
    'otdf_official/config/otdf/hopper.yaml': 'a3acb00ecee00ddafcb1b5d2a6ea5732acad34d4123dcc8f59b514ecf1898c32',
    'otdf_official/config/otdf/hopper_morph.yaml': '0a0649be9be27a29137937d25ce1cf0d5e132f89716b48f0a125ac729e585e66',
    'otdf_official/config/otdf/walker2d.yaml': '09403a50bcc4e6ac0d990864bbb0327d0aee9433cebb57155e71155c9a53fc63',
    'otdf_official/config/otdf/walker2d_gravity.yaml': '48e3e4127acc2d1d4a00f61075133a204e7dbf87cf019dd2af7958a619701816',
    'otdf_official/config/otdf/walker2d_morph.yaml': 'e004ba0eebb0796e40a80e1c2732d3a9204a05afced32f9e6d07aa1163807816',
    'otdf_official/envs/assets/hopper_morph.xml': '003f4d6399dddba9417f7af1b2579d766481e441d5e86f1469e9e87ac27fc7e5',
}
# Same files as checked out on the lab computer (CRLF), recorded in reproductions/otdf_setup manifests.
OTDF_REPO_CRLF_SHA256 = {
    'otdf_official/algo/OTDF.py': '93575496e5171207a89968b1dc66e3b7af363bc84a81a905bf04c242aff5a32a',
    'otdf_official/algo/utils.py': '8f78857a82f03d039d00ca6f85c6aadcd3fada9c0642aa0828cec0cc7df0c875',
    'otdf_official/make_otdf_cost.py': '831c67cb50c931a27be4bfd9a41e945acadbe22b0c338a625374f7aabc19bac5',
    'otdf_official/train_otdf.py': '19abd768440180b68917540f43440e65bd31e25b432f61dcc73d40f450747062',
    'otdf_official/info.py': 'fd4b47387a6e782c56fe63491fd93678d65a86340994452b142e2cda0952240b',
}
OTDF_HOPPER_MORPH_TARGET_SHA256 = '135d69d7ae3d20d179219b0ae4f489d77e29b651375a7e66f4a3f1b5d85cd7bb'
# Minimal d4rl 1.1 (this repository's src/d4rl/d4rl, used by the lab computer's OTDF reproduction; CR removed).
D4RL_SHA256 = {
    'd4rl/__init__.py': 'e553e6157ad01efd1e7bc43ffc6260211dd0d9792c5d47145bbd767948aca750',
    'd4rl/infos.py': '4460cbb9d49e05d5b494a94f922bac1ed3ee559dbf298fd14a658f70f8c86732',
    'd4rl/offline_env.py': 'e5c82d394ec71d569654ecfec110672dc82ef084c0ee09e646b7ce3f1333c834',
}
D4RL_REPO_CRLF_SHA256 = {
    'd4rl/__init__.py': '67fb5606d33c686d23315cb5dbd08a7ff9312f2be27e75599e1b7f23a4baa830',
    'd4rl/offline_env.py': '314261682aa0bc2035ef56aadaca394f1d0a5c70172ca823636e6b3d8c799713',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def clean(value):
    """JSON-safe copy: non-finite floats -> None, numpy scalars/arrays -> Python values."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist') and not isinstance(value, (str, bytes)):
        return clean(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def crlf_sha256(path):
    """SHA-256 of the file with LF converted to CRLF (to relate the LF copy to a Windows checkout)."""
    data = Path(path).read_bytes()
    assert b'\r\n' not in data, 'expected an LF file: %s' % rel(path)
    return sha256_bytes(data.replace(b'\n', b'\r\n'))


def array_digest(x):
    """SHA-256 of the C-contiguous little-endian bytes of a numpy array, with dtype and shape."""
    import numpy as np
    a = np.asarray(x)
    if a.dtype.byteorder == '>':
        a = a.astype(a.dtype.newbyteorder('<'))
    a = np.ascontiguousarray(a)
    return dict(sha256=sha256_bytes(a.tobytes()), dtype=str(a.dtype), shape=list(a.shape))


def seed_of(*parts):
    text = SEED_PREFIX+'|'+'|'.join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path):
    """Path for outputs: relative to code/ when inside it, '~/...' when inside the home directory, else the name."""
    p = Path(path)
    try:
        p = p.resolve()
    except Exception:
        pass
    for base, prefix in ((CODE, ''), (HOME, '~/')):
        try:
            r = p.relative_to(base.resolve())
            return prefix+str(r).replace('\\', '/')
        except ValueError:
            continue
    return p.name


def scrub(text):
    """Remove the home directory, user name and host name from free text (tracebacks, logs)."""
    text = str(text)
    home = str(HOME)
    if home and home != '/':
        text = text.replace(home, '~')
    for var in ('USER', 'LOGNAME'):
        name = os.environ.get(var)
        if name and len(name) >= 3:
            text = text.replace(name, '<USER>')
    try:
        import socket
        host = socket.gethostname()
        if host and len(host) >= 3:
            text = text.replace(host, '<HOST>')
    except Exception:
        pass
    return text


def load_tasks():
    spec = read_json(CODE/'tasks.json')
    tasks = {t['id']: t for t in spec['tasks']}
    return spec, tasks


def task_by_short(short):
    spec, tasks = load_tasks()
    for t in tasks.values():
        if t['short'] == short:
            return t
    raise KeyError(short)


def verify_frozen(code=CODE):
    """Every file listed in FROZEN.json has its recorded SHA-256; returns the list of checked files."""
    spec = read_json(Path(code)/'FROZEN.json')
    bad = [n for n, d in spec['files'].items() if sha256_file(Path(code)/n) != d]
    assert not bad, 'FROZEN hash mismatch: %s' % bad
    return sorted(spec['files'])


def verify_official(code=CODE):
    """Official OTDF files and the minimal d4rl copy have their expected bytes."""
    out = {}
    for name, digest in list(OTDF_OFFICIAL_SHA256.items())+list(D4RL_SHA256.items()):
        got = sha256_file(Path(code)/name)
        assert got == digest, 'official file changed: %s' % name
        out[name] = dict(sha256=got)
    for name, digest in list(OTDF_REPO_CRLF_SHA256.items())+list(D4RL_REPO_CRLF_SHA256.items()):
        got = crlf_sha256(Path(code)/name)
        assert got == digest, 'LF copy does not match the CRLF checkout recorded on the lab computer: %s' % name
        out[name]['crlf_checkout_sha256'] = got
    return out


def stats(values, ddof=1):
    n = len(values)
    if n == 0:
        return dict(n=0, mean=None, sd=None, se=None)
    mean = sum(values)/n
    sd = math.sqrt(sum((v-mean)**2 for v in values)/(n-ddof)) if n > ddof else 0.
    return dict(n=n, mean=mean, sd=sd, se=sd/math.sqrt(n) if n > 0 else None)


def mem_used_fraction():
    """System memory in use (1 - MemAvailable/MemTotal) from /proc/meminfo; None when unavailable."""
    try:
        info = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            k, v = line.split(':', 1)
            info[k.strip()] = float(v.strip().split()[0])
        return 1.-info['MemAvailable']/info['MemTotal']
    except Exception:
        return None


def spearman(x, y):
    """Spearman rank correlation with average ranks for ties; None when undefined."""
    n = len(x)
    if n < 2 or len(y) != n:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.]*n
        i = 0
        while i < n:
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]:
                j += 1
            for k in range(i, j+1):
                r[order[k]] = (i+j)/2.+1.
            i = j+1
        return r
    rx, ry = ranks(list(x)), ranks(list(y))
    mx, my = sum(rx)/n, sum(ry)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    sxx = sum((a-mx)**2 for a in rx)
    syy = sum((b-my)**2 for b in ry)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy/math.sqrt(sxx*syy)
