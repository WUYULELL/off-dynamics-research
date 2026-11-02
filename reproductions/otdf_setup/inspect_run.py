"""Read OTDF TensorBoard/console output without importing models or environments.

Example (from the project root):
  reproductions/otdf_train/Scripts/python.exe reproductions/otdf_setup/inspect_run.py \
    --run-dir reproductions/otdf_setup/runs/train_1M/otdf/\
walker2d-morph-srcdatatype-medium-tardatatype-medium/r1 --console-log train.log

Supply --exit-code only after observing the real launcher exit code. The official
trainer emits no completion marker, and a run shorter than 5000 policy steps
emits no policy training scalar. CVAE steps are never policy steps here.
This command writes only JSON to stdout and does not create or alter run files.
"""

import argparse
import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys


POLICY_TAGS = (
    "train/adv", "train/value", "train/q1",
    "test/source_return", "test/target_return",
)
# TensorBoardX serializes spaces in these official source-code tags as
# underscores. Accept the unsanitized spelling too, without counting twice.
TAG_ALIASES = {
    "test/source return": "test/source_return",
    "test/target return": "test/target_return",
    "train/VAE loss": "train/VAE_loss",
}
EVAL_PATTERN = re.compile(
    r"\[(\d+)\]\s+Evaluation over\s+(\d+)\s+episodes:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|nan|inf(?:inity)?))",
    re.IGNORECASE,
)


def read_text(path):
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data[:100]:
        return data.decode("utf-16-le", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("gb18030", errors="replace")


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def scalar_point(value, step, wall_time=None):
    value = float(value)
    return {
        "step": int(step),
        "value": value if math.isfinite(value) else None,
        "finite": math.isfinite(value),
        "nonfinite_value": None if math.isfinite(value) else str(value),
        "wall_time_utc": utc(wall_time) if wall_time is not None else None,
    }


def summarize_events(events):
    ordered = sorted(events, key=lambda e: (e.wall_time, e.step))
    bad = [scalar_point(e.value, e.step, e.wall_time)
           for e in ordered if not math.isfinite(float(e.value))]
    steps = [e.step for e in ordered]
    return {
        "count": len(ordered),
        "all_finite": not bad if ordered else None,
        "nonfinite_count": len(bad),
        "nonfinite_examples": bad[:20],
        "duplicate_step_count": len(steps) - len(set(steps)),
        "step_regressions": sum(b < a for a, b in zip(steps, steps[1:])),
        "latest": scalar_point(ordered[-1].value, ordered[-1].step,
                               ordered[-1].wall_time) if ordered else None,
    }


def parse_console(text, eval_freq):
    records = []
    per_count = {}
    for match in EVAL_PATTERN.finditer(text):
        count, episodes, value = int(match[1]), int(match[2]), float(match[3])
        occurrence = per_count.get(count, 0)
        per_count[count] = occurrence + 1
        # Official train_otdf.py always prints source first, target second.
        record = scalar_point(value, count * eval_freq)
        record.update({"eval_count": count, "episodes": episodes,
                       "domain": "source" if occurrence % 2 == 0 else "target"})
        records.append(record)
    errors = [line.strip() for line in text.splitlines() if re.search(
        r"Traceback \(most recent call last\)|CUDA out of memory|"
        r"(?:RuntimeError|MemoryError|ValueError|AssertionError|ImportError|"
        r"ModuleNotFoundError|OSError|FloatingPointError):|"
        r"Training failed with exit code", line)]
    return {
        "evaluation_records": records,
        "duplicate_evaluation_counts": [key for key, count in per_count.items() if count > 2],
        "unpaired_evaluation_counts": [key for key, count in per_count.items() if count % 2],
        "error_lines": errors[-20:],
    }


def references(repo, env):
    # Read constants as data; never import info.py or any environment package.
    source = repo / "info.py"
    constants = {}
    for node in ast.parse(read_text(source)).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("REF_MIN_SCORE", "REF_MAX_SCORE"):
                    constants[target.id] = ast.literal_eval(node.value)
    return {
        "env": env,
        "ref_min": constants["REF_MIN_SCORE"][env],
        "ref_max": constants["REF_MAX_SCORE"][env],
        "source": str(source),
        "formula": "100 * (raw_return - ref_min) / (ref_max - ref_min)",
        "clipped": False,
    }


def normalized(point, refs):
    if point is None:
        return None
    result = dict(point)
    result["value"] = ((point["value"] - refs["ref_min"]) * 100 /
                       (refs["ref_max"] - refs["ref_min"])) if point["finite"] else None
    return result


def speed_estimate(events, planned, current_step):
    ordered = sorted(events, key=lambda e: (e.wall_time, e.step))
    if len(ordered) < 2:
        return None
    # A restart/reset invalidates a simple extrapolation across that boundary.
    last_reset = max([i for i in range(1, len(ordered))
                      if ordered[i].step <= ordered[i - 1].step] or [0])
    recent = ordered[last_reset:][-6:]
    if len(recent) < 2:
        return None
    seconds = recent[-1].wall_time - recent[0].wall_time
    steps = recent[-1].step - recent[0].step
    if seconds <= 0 or steps <= 0:
        return None
    rate = steps / seconds
    remaining = max(0, planned - current_step) / rate if planned is not None else None
    return {"steps_per_second": rate, "window_seconds": seconds,
            "window_policy_steps": steps,
            "from_step": recent[0].step, "to_step": recent[-1].step,
            "estimated_remaining_seconds_at_last_observation": remaining,
            "assumption": "Recent throughput continues without interruption; excludes prior OT/CVAE time."}


def option_value(arguments, key):
    try:
        return arguments[arguments.index(key) + 1]
    except (ValueError, IndexError):
        return None


def inspect(args):
    run = args.run_dir.resolve()
    repo = Path(__file__).resolve().parent.parent / "OTDF"
    notes = ["Only recorded scalars/returns are checked; finite logs do not prove every model parameter or loss is finite."]
    errors = []
    invocation_path = args.invocation
    if invocation_path is None and run.name.startswith("r") and run.name[1:].isdigit():
        for parent in run.parents:
            if parent.parent.name == "runs":
                candidate = parent.parent / (parent.name + "_seed_" + run.name[1:] + ".invocation.json")
                if candidate.is_file():
                    invocation_path = candidate
                break
    invocation = json.loads(read_text(invocation_path)) if invocation_path else {}
    official_args = invocation.get("official_arguments", [])
    env = args.env or option_value(official_args, "--env") or run.parent.name.split("-srcdatatype-")[0]
    planned = args.planned_steps if args.planned_steps is not None else invocation.get("policy_steps")
    if planned is not None:
        planned = int(planned)
    eval_freq = args.eval_freq
    if eval_freq is None:
        config_path = repo / "config" / "otdf" / (env.replace("-", "_") + ".yaml")
        match = re.search(r"(?m)^eval_freq:\s*(\d+)", read_text(config_path)) if config_path.is_file() else None
        eval_freq = int(match[1]) if match else 10000
        if match is None:
            notes.append("No matching YAML eval_freq found; assumed official default 10000 for console step mapping.")

    tb_dir = run if any(run.glob("events.out.tfevents.*")) else run / "tb"
    event_files = sorted(tb_dir.glob("events.out.tfevents.*"))
    scalar_events = {}
    if event_files:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            accumulator = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0}, purge_orphaned_data=False)
            accumulator.Reload()
            for raw_tag in accumulator.Tags().get("scalars", []):
                tag = TAG_ALIASES.get(raw_tag, raw_tag)
                if tag in scalar_events:
                    raise ValueError("Ambiguous duplicate scalar tag after alias mapping: " + tag)
                scalar_events[tag] = accumulator.Scalars(raw_tag)
        except Exception as exc:
            errors.append("TensorBoard read failed: {}: {}".format(type(exc).__name__, exc))
    metrics = {tag: summarize_events(events) for tag, events in sorted(scalar_events.items())}
    console = []
    for path in args.console_log:
        try:
            parsed = parse_console(read_text(path), eval_freq)
            parsed["path"] = str(path.resolve())
            console.append(parsed)
        except OSError as exc:
            errors.append("Console read failed: {}".format(exc))
    console_records = [record for item in console for record in item["evaluation_records"]]

    last_returns = {}
    for domain in ("source", "target"):
        tag = "test/{}_return".format(domain)
        event_point = metrics.get(tag, {}).get("latest")
        console_points = [record for record in console_records if record["domain"] == domain]
        console_point = max(console_points, key=lambda p: p["step"]) if console_points else None
        if event_point is not None and (console_point is None or event_point["step"] >= console_point["step"]):
            last_returns[domain] = dict(event_point, origin="tensorboard")
        else:
            last_returns[domain] = dict(console_point, origin="console") if console_point else None

    logged_steps = [event.step for tag in POLICY_TAGS for event in scalar_events.get(tag, [])]
    logged_steps += [point["step"] for point in console_records]
    last_logged_step = max(logged_steps) if logged_steps else None
    finite_count = sum(item["count"] for item in metrics.values()) + len(console_records)
    nonfinite_count = sum(item["nonfinite_count"] for item in metrics.values()) + sum(not p["finite"] for p in console_records)
    successful_exit = args.exit_code == 0 if args.exit_code is not None else None
    inferred_complete_step = planned if successful_exit and planned is not None else None
    current_step = max([s for s in (last_logged_step, inferred_complete_step) if s is not None], default=None)
    if successful_exit is True:
        status = "process_exited_successfully"
    elif successful_exit is False:
        status = "process_exited_with_error"
    elif planned is not None and last_logged_step is not None and last_logged_step >= planned:
        status = "planned_steps_observed_exit_unconfirmed"
    elif not run.exists():
        status = "run_directory_not_created"
    else:
        status = "in_progress_or_stopped_exit_unknown"
    if successful_exit is None:
        notes.append("Official logs have no completion marker. Process liveness/exit status is unknown; --exit-code requires the actual observed launcher result.")
    if planned is not None and planned < 5000:
        notes.append("This short run has no policy training scalars; CVAE steps up to 9001 do not measure policy progress.")

    timing = None
    for tag in ("test/target_return", "train/value", "train/q1", "train/adv"):
        timing = speed_estimate(scalar_events.get(tag, []), planned, current_step or 0)
        if timing:
            timing["tag"] = tag
            timing["includes_regular_evaluation_overhead"] = tag == "test/target_return"
            break
    refs = None
    target_normalized = None
    if env.startswith("walker2d"):
        try:
            refs = references(repo, env)
            target_normalized = normalized(last_returns["target"], refs)
        except (KeyError, OSError, ValueError) as exc:
            errors.append("Target normalization unavailable: {}".format(exc))
    else:
        notes.append("Walker2d normalization only; no score conversion applied to the inferred task.")
    return {
        "schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run), "env": env, "status": status,
        "invocation_path": str(invocation_path.resolve()) if invocation_path else None,
        "planned_policy_steps": planned, "observed_process_exit_code": args.exit_code,
        "last_logged_policy_step": last_logged_step,
        "last_policy_step": current_step,
        "last_policy_step_basis": "successful official loop exit and planned steps" if inferred_complete_step is not None else "logged policy metrics/evaluations only",
        "cvae_last_logged_step": metrics["train/VAE_loss"]["latest"]["step"] if metrics.get("train/VAE_loss", {}).get("latest") else None,
        "cvae_latest_loss": metrics.get("train/VAE_loss", {}).get("latest"),
        "finite_check": {"observations": finite_count, "nonfinite_observations": nonfinite_count,
                         "all_recorded_values_finite": nonfinite_count == 0 if finite_count else None},
        "latest_source_raw_return": last_returns["source"],
        "latest_target_raw_return": last_returns["target"],
        "latest_target_normalized_score": target_normalized,
        "normalization": refs, "timing": timing,
        "tensorboard_files": [str(path) for path in event_files],
        "scalar_tags": metrics, "console_logs": console,
        "notes": notes, "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Official run directory, normally .../r1")
    parser.add_argument("--console-log", type=Path, action="append", default=[], help="Optional stdout/stderr log; repeatable")
    parser.add_argument("--invocation", type=Path, help="Launcher invocation JSON (auto-discovered when possible)")
    parser.add_argument("--env", help="Override task name, e.g. walker2d-morph")
    parser.add_argument("--planned-steps", type=int, help="Override policy step budget; otherwise read invocation")
    parser.add_argument("--eval-freq", type=int, help="Override console count-to-step conversion; otherwise read YAML")
    parser.add_argument("--exit-code", type=int, help="Only supply the actual observed launcher process exit code")
    args = parser.parse_args()
    try:
        result = inspect(args)
        code = 2 if result["errors"] else (1 if result["finite_check"]["nonfinite_observations"] or args.exit_code not in (None, 0) else 0)
    except Exception as exc:
        result = {"schema_version": 1, "errors": ["{}: {}".format(type(exc).__name__, exc)]}
        code = 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
