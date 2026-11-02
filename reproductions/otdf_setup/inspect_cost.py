"""Read-only OT cost diagnostics; no torch/JAX/Gym imports or GPU use.

The sampled batches are fixed-seed diagnostics, NOT the real first training
batch: VAE pretraining and other code can advance the training NumPy RNG.
Only stdout JSON is emitted. No files, models, environments or processes are
created/changed. Run after cost generation has closed its HDF5 file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

try:
    import h5py
    import numpy as np
except ImportError as exc:
    print(json.dumps({"status": "dependency_missing", "error": str(exc),
                      "required": ["numpy", "h5py"]}, ensure_ascii=False))
    sys.exit(2)


QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def finite_summary(values: np.ndarray) -> dict:
    values = np.asarray(values).reshape(-1)
    finite = values[np.isfinite(values)]
    unique_count = int(np.unique(finite).size)
    return {
        "count": int(values.size),
        "dtype": str(values.dtype),
        "finite_count": int(finite.size),
        "finite_fraction": float(finite.size / values.size) if values.size else None,
        "nan_count": int(np.isnan(values).sum()),
        "positive_inf_count": int(np.isposinf(values).sum()),
        "negative_inf_count": int(np.isneginf(values).sum()),
        "zero_count": int((values == 0).sum()),
        "positive_count": int((values > 0).sum()),
        "min_finite": float(finite.min()) if finite.size else None,
        "max_finite": float(finite.max()) if finite.size else None,
        "quantiles_finite": {
            str(q): float(value)
            for q, value in zip(QUANTILES, np.quantile(finite, QUANTILES))
        } if finite.size else {},
        "unique_finite_count": unique_count,
        "unique_fraction_of_finite": float(unique_count / finite.size) if finite.size else None,
    }


def batch_diagnostic(cost: np.ndarray, indices: np.ndarray, proportion: float) -> dict:
    # ReplayBuffer.sample() uses torch.FloatTensor(self.cost[ind]): float32.
    values = np.asarray(cost[indices], dtype=np.float32)
    keep = int(values.size * proportion)
    with np.errstate(all="ignore"):
        lo = np.min(values)
        hi = np.max(values)
        width = np.float32(hi - lo)
        # Deliberately mirror the published implementation WITHOUT an epsilon.
        normalized = (values - hi) / width
        all_weights = np.exp(normalized)

    all_finite = bool(np.isfinite(normalized).all())
    if all_finite:
        # For ties, different top-k indices have identical scalar weights.
        selected = np.sort(normalized)[-keep:]
        weights = np.exp(selected)
        selected_summary = finite_summary(weights)
        selected_status = "finite_top_k_values; tie indices are not reproduced"
    else:
        # Do not claim NumPy's NaN ordering matches torch.topk's NaN ordering.
        selected_summary = None
        selected_status = "not_simulated_due_to_nonfinite_normalized_values"

    return {
        "sample_count": int(values.size),
        "keep_count": keep,
        "indices_head": [int(x) for x in indices[:16]],
        "indices_sha256_int64_le": hashlib.sha256(
            np.asarray(indices, dtype="<i8").tobytes()
        ).hexdigest(),
        "cost_float32": finite_summary(values),
        "range_float32": float(width) if np.isfinite(width) else None,
        "range_is_zero": bool(width == 0),
        "range_is_finite": bool(np.isfinite(width)),
        "normalized_nonfinite_count": int((~np.isfinite(normalized)).sum()),
        "weights_before_topk_nonfinite_count": int((~np.isfinite(all_weights)).sum()),
        "retained_weights": selected_summary,
        "retained_weights_status": selected_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cost-path", type=Path,
        default=Path(__file__).resolve().parent.parent / "OTDF" / "costlogs" /
        "walker2d-morph-srcdatatype-medium-tardatatype-medium.hdf5",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--proportion", type=float, default=0.8)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    result = {
        "path": str(args.cost_path.resolve()),
        "mode": "read_only_cpu_numpy_h5py",
        "sampling_label": "fixed_seed_diagnostic_NOT_actual_first_training_batch",
        "sampling_note": (
            "Uses an independent NumPy RandomState(seed). The real training RNG "
            "is advanced by CVAE sampling and possibly other code; no complete "
            "training RNG replay is claimed. No model/environment is instantiated."
        ),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "diagnostic_batches": args.batches,
        "proportion": args.proportion,
    }
    try:
        if args.batch_size < 1 or args.batches < 1:
            raise ValueError("batch-size and batches must be positive")
        if not 0 < args.proportion <= 1 or int(args.batch_size * args.proportion) < 1:
            raise ValueError("proportion must retain at least one sample")
        if not args.cost_path.is_file():
            result.update(status="not_ready", reason="cost_file_missing")
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 2
        before = args.cost_path.stat()
        with h5py.File(args.cost_path, "r") as handle:
            if "cost" not in handle:
                raise ValueError("HDF5 has no cost dataset")
            dataset = handle["cost"]
            result["hdf5_shape"] = list(dataset.shape)
            result["hdf5_dtype"] = str(dataset.dtype)
            if len(dataset.shape) != 1:
                raise ValueError("Expected a one-dimensional cost vector matching official output")
            cost = np.asarray(dataset[:])
        after = args.cost_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            result.update(status="not_ready", reason="cost_file_changed_during_read")
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 2
        if not cost.size or cost.dtype.kind not in "fiu":
            raise ValueError("Expected a nonempty real numeric cost vector")
        result["file_size_bytes"] = after.st_size
        result["expected_count"] = args.expected_count
        result["count_matches_expected"] = (
            bool(cost.size == args.expected_count) if args.expected_count is not None else None
        )
        result["cost_stored"] = finite_summary(cost)
        with np.errstate(all="ignore"):
            cost32 = cost.astype(np.float32)
        result["cost_training_float32"] = finite_summary(cost32)
        random = np.random.RandomState(args.seed)
        first = None
        zero_ranges = nonfinite_ranges = bad_normalization = 0
        min_range = max_range = None
        first_bad = None
        for batch_number in range(args.batches):
            indices = random.randint(0, cost.size, size=args.batch_size)
            diagnostic = batch_diagnostic(cost32, indices, args.proportion)
            if first is None:
                first = diagnostic
            zero_ranges += int(diagnostic["range_is_zero"])
            nonfinite_ranges += int(not diagnostic["range_is_finite"])
            bad_normalization += int(diagnostic["normalized_nonfinite_count"] > 0)
            width = diagnostic["range_float32"]
            if width is not None:
                min_range = width if min_range is None else min(min_range, width)
                max_range = width if max_range is None else max(max_range, width)
            if first_bad is None and diagnostic["normalized_nonfinite_count"]:
                first_bad = {"zero_based_diagnostic_batch": batch_number, **diagnostic}
        result["first_fixed_seed_diagnostic_batch"] = first
        result["batch_summary"] = {
            "range_zero_batches": zero_ranges,
            "range_nonfinite_batches": nonfinite_ranges,
            "normalization_or_weight_nonfinite_batches": bad_normalization,
            "min_finite_range": min_range,
            "max_finite_range": max_range,
            "first_bad_diagnostic_batch": first_bad,
        }
        problems = []
        if result["count_matches_expected"] is False:
            problems.append("cost_length_does_not_match_expected_source_count")
        if result["cost_training_float32"]["finite_fraction"] != 1.0:
            problems.append("nonfinite_cost_after_training_float32_conversion")
        if result["cost_training_float32"]["unique_finite_count"] < 2:
            problems.append("global_cost_degenerate; official_minmax_normalization_will_divide_by_zero")
        if bad_normalization:
            problems.append("sampled_diagnostic_batches_produce_nonfinite_normalization_or_weights")
        result["problems"] = problems
        result["status"] = "numeric_failure" if problems else "checks_passed_for_fixed_seed_diagnostic"
        result["interpretation"] = (
            "Passing these checks does not prove OT convergence, correct coupling, "
            "the safety of every future batch, or paper-level reproduction. Float64 "
            "HDF5 storage does not undo quantization during float16 OT computation."
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if problems else 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        result.update(status="read_or_validation_error", error_type=type(exc).__name__, error=str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
