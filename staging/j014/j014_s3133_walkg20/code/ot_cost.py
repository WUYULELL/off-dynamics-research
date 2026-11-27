"""OT cost of one task with the official solve_ot (run by an OT-capable Python: jax + ott-jax + h5py + numpy).

Same procedure as reproductions/otdf_setup/compute_ot_cpu.py: the solve_ot function is extracted with AST from
the official make_otdf_cost.py (commit 41e2e62, hash checked) and jit-compiled; settings stay official (float16
embeddings, cosine cost, scale_cost='max_cost', Sinkhorn threshold 1e-9 / max_iterations 100, -ot_costs);
source rows in chunks of 10000 against the whole target set; results gathered with .tolist() and stored as the
official float64 array 'cost' (gzip) in hdf5. The only change to the official loop is that no empty final chunk
is submitted when the row count is a multiple of 10000. Standalone: standard library + numpy + h5py + jax + ott.

usage: python ot_cost.py <request.json>
request: {"task", "source_npy", "target_npy", "source_sas_sha256", "target_sas_sha256", "official_make_cost",
          "official_make_cost_sha256", "out_hdf5", "out_manifest"}
"""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def version(name):
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        try:
            return getattr(importlib.import_module(name.replace('-', '_')), '__version__', None)
        except Exception:
            return None


def main():
    req = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    out = Path(req['out_hdf5'])
    meta_path = Path(req['out_manifest'])
    partial = out.with_name(out.name+'.partial')
    for p in (out, meta_path):
        if p.exists():
            raise FileExistsError('refusing to overwrite %s' % p.name)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import numpy as np
    import h5py
    import jax
    import jax.numpy as jnp
    import ott

    official = Path(req['official_make_cost'])
    assert sha256_file(official) == req['official_make_cost_sha256'], 'official make_otdf_cost.py changed'
    tree = ast.parse(official.read_text(encoding='utf-8'), filename=str(official))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'solve_ot']
    if len(functions) != 1:
        raise RuntimeError('expected exactly one official solve_ot function')
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {'jnp': jnp, 'ott': ott}
    exec(compile(ast.fix_missing_locations(module), str(official), 'exec'), namespace)
    batch_solve = jax.jit(namespace['solve_ot'])

    def array_sha(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    src = np.load(req['source_npy'], allow_pickle=False)
    tar = np.load(req['target_npy'], allow_pickle=False)
    assert src.dtype == np.float32 and tar.dtype == np.float32 and src.ndim == tar.ndim == 2 and src.shape[1] == tar.shape[1]
    assert array_sha(src) == req['source_sas_sha256'], 'source [s, a, s\'] array differs from the export'
    assert array_sha(tar) == req['target_sas_sha256'], 'target [s, a, s\'] array differs from the export'

    started = time.monotonic()
    values = []
    chunks = 0
    for start in range(0, src.shape[0], 10000):       # official: range(src_num // 10000 + 1) (empty tail skipped)
        stop = min(start+10000, src.shape[0])
        part = jax.device_get(batch_solve(src[start:stop], tar))
        part = np.asarray(part)
        if part.shape != (stop-start,) or not np.isfinite(part.astype(np.float64)).all():
            raise RuntimeError('invalid OT output in rows %d-%d; no cost file saved' % (start, stop))
        values += part.tolist()                         # official list -> np.array (float64) storage
        chunks += 1
        elapsed = time.monotonic()-started
        if chunks == 1 or chunks % 10 == 0 or stop == src.shape[0]:
            print('[ot %s] %d / %d rows, %.1f s elapsed, %.1f s left (est.)' % (
                req['task'], stop, src.shape[0], elapsed, elapsed*(src.shape[0]-stop)/stop), flush=True)
    cost = np.array(values)
    assert cost.shape == (src.shape[0],) and cost.dtype == np.float64
    with h5py.File(str(partial), 'w') as handle:
        handle.create_dataset('cost', data=cost, compression='gzip')
    os.replace(str(partial), str(out))
    meta = dict(schema=1, task=req['task'], created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                source_rows=int(src.shape[0]), target_rows=int(tar.shape[0]), columns=int(src.shape[1]),
                source_sas_sha256=req['source_sas_sha256'], target_sas_sha256=req['target_sas_sha256'],
                cost_file=out.name, cost_sha256=sha256_file(out), cost_bytes=out.stat().st_size,
                cost_dtype=str(cost.dtype), cost_min=float(cost.min()), cost_max=float(cost.max()),
                cost_is_negative_matching_cost=True, chunks=chunks, elapsed_seconds=time.monotonic()-started,
                solve_ot_from='otdf_official/make_otdf_cost.py (AST, unchanged)',
                official_make_cost_sha256=req['official_make_cost_sha256'],
                settings=dict(dtype='float16', cost='cosine', scale_cost='max_cost', threshold=1e-9,
                              max_iterations=100, chunk=10000),
                deviations=['the empty final chunk is not submitted when the row count is a multiple of 10000 '
                            '(official: range(src_num // 10000 + 1)); every non-empty chunk and the target set are unchanged',
                            'solve_ot taken from the official file by AST (no unrelated torch / gym / d4rl imports)'],
                python=platform.python_version(), platform=platform.system()+' '+platform.machine(),
                backend=jax.default_backend(), devices=[str(d.platform) for d in jax.devices()],
                packages={n: version(n) for n in ('jax', 'jaxlib', 'ott-jax', 'jaxopt', 'numpy', 'scipy', 'h5py')},
                xla_flags=os.environ.get('XLA_FLAGS'))
    tmp = meta_path.with_name(meta_path.name+'.partial')
    tmp.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    os.replace(str(tmp), str(meta_path))
    print('[ot %s] saved %s (%d rows)' % (req['task'], out.name, len(cost)), flush=True)


if __name__ == '__main__':
    main()
