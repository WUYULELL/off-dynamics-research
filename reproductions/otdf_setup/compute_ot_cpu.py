"""Run OT only when explicitly launched; never imports the official script as a module.

Extracts its solve_ot function with AST, keeping that function unchanged. The only
loop correction is not submitting an empty final chunk for a multiple of 10000.
"""
import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

from export_ot_inputs import DEFAULT_INPUTS, REPO, TASK, file_record, sha256, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    args = parser.parse_args()
    out = REPO / "costlogs" / (TASK + ".hdf5")
    meta_path = out.with_suffix(".manifest.json")
    temporary = out.with_suffix(".hdf5.partial")
    for path in (out, meta_path, temporary):
        if path.exists():
            raise FileExistsError("Refusing to overwrite existing output: {}".format(path))
    manifest = verify_inputs(args.inputs)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import h5py
    import jax
    import jax.numpy as jnp
    import numpy as np
    import ott

    official_path = REPO / "make_otdf_cost.py"
    tree = ast.parse(official_path.read_text(encoding="utf-8"), filename=str(official_path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve_ot"]
    if len(functions) != 1:
        raise RuntimeError("Expected exactly one official solve_ot function")
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"jnp": jnp, "ott": ott}
    exec(compile(ast.fix_missing_locations(module), str(official_path), "exec"), namespace)
    batch_solve = jax.jit(namespace["solve_ot"])
    src = np.load(manifest["features"]["source"]["path"], mmap_mode="r", allow_pickle=False)
    tar = np.load(manifest["features"]["target"]["path"], mmap_mode="r", allow_pickle=False)
    for array, key in [(src, "source"), (tar, "target")]:
        expected = manifest["features"][key]
        if list(array.shape) != [expected["rows"], expected["columns"]] or str(array.dtype) != expected["dtype"]:
            raise RuntimeError("Array shape/dtype mismatch: " + key)
    print("CPU OT begins only because Cost was explicitly requested.", flush=True)
    print("Official float16/cosine/max_cost/Sinkhorn(1e-9,100), chunk=10000; Windows JAX differs from paper.", flush=True)
    start_time = time.monotonic()
    values = []
    for start in range(0, src.shape[0], 10000):
        stop = min(start + 10000, src.shape[0])
        part = jax.device_get(batch_solve(src[start:stop], tar))
        if part.shape != (stop - start,) or not np.isfinite(part).all():
            raise RuntimeError("Invalid OT output; no cost file saved. Investigate float16/CPU numerics before proceeding")
        values += part.tolist()  # Keep official list -> np.array (float64) storage.
        elapsed = time.monotonic() - start_time
        print("Completed {} / {} transitions; elapsed {:.1f}s; estimated remaining {:.1f}s".format(
            stop, src.shape[0], elapsed, elapsed * (src.shape[0] - stop) / stop), flush=True)
    cost = np.array(values)
    if cost.shape != (manifest["features"]["source"]["rows"],):
        raise RuntimeError("Cost length differs from source row order")
    # Re-check original/feature files before publishing a cost generated for them.
    verify_inputs(args.inputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(temporary), "x") as handle:
        handle.create_dataset("cost", data=cost, compression="gzip")
    metadata = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "task": TASK,
        "inputs_manifest_sha256": sha256(args.inputs / "manifest.json"),
        "source_rows": int(src.shape[0]), "target_rows": int(tar.shape[0]),
        "feature_order_preserved": True, "cost_is_negative": True,
        "solve_ot_source_sha256": sha256(official_path),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ["jax", "jaxlib", "ott-jax", "jaxopt", "numpy", "scipy", "h5py"]},
        "backend": "cpu", "elapsed_seconds": time.monotonic() - start_time,
        "settings": {"dtype": "float16", "cost": "cosine", "scale_cost": "max_cost",
                     "threshold": 1e-9, "max_iterations": 100, "chunk": 10000},
        "deviations": ["Windows CPU JAX/jaxlib 0.4.20 instead of authors' 0.4.9; numerical equivalence unverified",
                       "Uses official solve_ot AST without unrelated Torch/Gym imports",
                       "Skip the empty tail chunk only; non-empty chunks and target data unchanged"],
        "cost_file": file_record(temporary),
    }
    metadata["cost_file"]["path"] = str(out.resolve())
    with meta_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    temporary.rename(out)  # Windows rename refuses to overwrite an existing file.
    print("Saved official-format cost: {}".format(out))


if __name__ == "__main__":
    main()
