"""Validate a fixed OTDF study and aggregate completed seeds, without training.

Usage: python -B aggregate_seeds.py --study study_five_seeds.json --output results
       python -B aggregate_seeds.py --study study_five_seeds.json --output results --validate-seed 2

Only summary.json, per_seed.csv, and SUMMARY.md are written in --output. Source
logs, manifests, datasets, and checkpoints are read-only. No torch/Gym/JAX import.
Exit 0: valid complete/partial study (and requested seed complete, if specified).
Exit 1: invalid evidence, or --validate-seed has not completed. Exit 2: tool error.
"""

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile


FINAL_STEP = 1000000
LAST5_STEPS = [960000, 970000, 980000, 990000, 1000000]
EXPECTED_SEEDS = [1, 2, 3, 4, 5]
REQUIRED_TAGS = ("train/VAE_loss", "train/adv", "train/value", "train/q1",
                 "test/source_return", "test/target_return")


class InvalidEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def no_constant(value):
    raise InvalidEvidence("JSON contains a non-finite constant: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=no_constant)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(value, base):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def path_equal(left, right, base):
    return os.path.normcase(str(resolve(left, base))) == os.path.normcase(str(resolve(right, base)))


def arguments(values):
    """Parse this trainer's CLI as data; reject duplicate/ambiguous options."""
    result = {}
    flags = {"--weight", "--noreg", "--save-model"}
    index = 0
    while index < len(values):
        key = values[index]
        require(isinstance(key, str) and key.startswith("--"), "Invalid official argument: " + str(key))
        require(key not in result, "Duplicate official argument: " + key)
        if key in flags:
            result[key] = True
            index += 1
        else:
            require(index + 1 < len(values) and not str(values[index + 1]).startswith("--"), "Missing value: " + key)
            result[key] = values[index + 1]
            index += 2
    return result


def without_run_arguments(values):
    result = []
    index = 0
    while index < len(values):
        if values[index] in ("--seed", "--dir"):
            index += 2
        else:
            result.append(values[index])
            index += 1
    return result


def nested(data, keys):
    for key in keys:
        if not isinstance(data, dict) or str(key) not in data:
            return None
        data = data[str(key)]
    return data


def iso_time(value):
    # CPython 3.8 supports microseconds; Windows status has seven decimals.
    value = re.sub(r"(\.\d{6})\d+(?=Z|[+-]\d\d:\d\d$)", r"\1", value)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def check_file(record, base, label):
    path = resolve(record["path"], base)
    require(path.is_file(), label + " missing: " + str(path))
    sha = digest(path)
    require(sha == record["sha256"].lower(), label + " SHA256 mismatch: " + str(path))
    return path


def validate_study(study, base):
    require(study.get("schema_version") == 1, "Unsupported study schema")
    require(study["expected_seeds"] == EXPECTED_SEEDS, "Expected exactly seeds 1,2,3,4,5")
    rows = study["seeds"]
    require(len(rows) == 5 and sorted(row["seed"] for row in rows) == EXPECTED_SEEDS, "Duplicate/missing/unexpected seed in study")
    for key in ("run_dir", "invocation_path", "inspection_path"):
        paths = [os.path.normcase(str(resolve(row[key], base))) for row in rows]
        require(len(set(paths)) == 5, "Different seeds reference the same " + key)
    require(study["policy_steps"] == FINAL_STEP and study["cvae_steps"] == 10000, "Study training budget differs from agreed protocol")
    require(study["eval_episodes"] == 10 and study["eval_frequency"] == 10000, "Study evaluation protocol differs")
    metrics = study["metrics"]
    require(metrics["primary"] == "Final" and metrics["final_step"] == FINAL_STEP, "Incorrect primary metric")
    require(metrics["last5_steps"] == LAST5_STEPS and metrics["std_ddof"] == 1, "Incorrect Last5/sample-SD definition")
    common = arguments(study["common_official_arguments"])
    require("--seed" not in common and "--dir" not in common, "Common arguments must exclude seed and output directory")
    require(int(common["--steps"]) == FINAL_STEP and common["--policy"] == "OTDF", "Common policy/budget mismatch")
    expected_task = "{}-srcdatatype-{}-tardatatype-{}".format(common["--env"], common["--srctype"], common["--tartype"])
    require(study["task"] == expected_task, "Task differs from common official arguments")

    inputs_path = check_file(study["inputs_manifest"], base, "Inputs manifest")
    cost_path = check_file(study["cost_manifest"], base, "Cost manifest")
    inputs, cost = read_json(inputs_path), read_json(cost_path)
    require(inputs["commit"] == study["official_commit"], "Official revision differs from exported inputs")
    require(inputs["task"] == study["task"] == cost["task"], "Data/cost task mismatch")
    require(cost["inputs_manifest_sha256"] == study["inputs_manifest"]["sha256"], "Cost is bound to different input manifest")
    # Check all locked source/data files and the actual OT cost without loading
    # any data/model framework. Exported feature arrays are not training inputs.
    checked = [str(check_file(record, base, "Locked source/data")) for record in inputs["original_files"]]
    checked.append(str(check_file(cost["cost_file"], base, "OT cost")))
    repo = Path(__file__).resolve().parent.parent / "OTDF"
    constants = {}
    for node in ast.parse((repo / "info.py").read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in ("REF_MIN_SCORE", "REF_MAX_SCORE"):
                    constants[name.id] = ast.literal_eval(node.value)
    env = common["--env"]
    minimum, maximum = constants["REF_MIN_SCORE"][env], constants["REF_MAX_SCORE"][env]
    require(study["normalization"]["ref_min"] == minimum and study["normalization"]["ref_max"] == maximum, "Normalization differs from official info.py")
    require(maximum > minimum, "Invalid normalization denominator")
    config = (repo / "config" / "otdf" / (env.replace("-", "_") + ".yaml")).read_text(encoding="utf-8-sig")
    require(int(re.search(r"(?m)^eval_freq:\s*(\d+)", config)[1]) == study["eval_frequency"], "Locked YAML evaluation frequency mismatch")
    tree = ast.parse((repo / "train_otdf.py").read_text(encoding="utf-8-sig"))
    evaluator = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "eval_policy")
    defaults = dict(zip([a.arg for a in evaluator.args.args][-len(evaluator.args.defaults):], evaluator.args.defaults))
    require(ast.literal_eval(defaults["eval_episodes"]) == study["eval_episodes"], "Official evaluation episode count mismatch")
    return {"common": common, "repo": repo, "checked_files": checked}


def validate_invocation(study, row, invocation, base, context):
    require(type(invocation.get("seed")) is int and invocation["seed"] == row["seed"], "Invocation seed mismatch")
    require(invocation.get("stage") == "Train", "Selected invocation is not Train")
    require(invocation.get("policy_steps") == study["policy_steps"] and invocation.get("cvae_steps") == study["cvae_steps"], "Invocation training budget mismatch")
    values = invocation["official_arguments"]
    options = arguments(values)
    require(without_run_arguments(values) == study["common_official_arguments"], "Invocation hyperparameters differ from fixed common arguments")
    require(int(options["--seed"]) == row["seed"], "CLI seed mismatch")
    expected = resolve(options["--dir"], base) / options["--policy"].lower() / study["task"] / ("r" + str(row["seed"]))
    require(path_equal(expected, row["run_dir"], base), "CLI output directory differs from selected run")
    require(path_equal(invocation["official_entrypoint"], context["repo"] / "train_otdf.py", base), "Wrong official entrypoint")
    require(path_equal(invocation["cost_manifest"], study["cost_manifest"]["path"], base), "Invocation points to a different cost manifest")


def validate_inspection(study, row, report, validation, base):
    require(report.get("observed_process_exit_code") == 0, "Inspection does not confirm exit 0")
    require(report.get("errors") == [], "Inspection has errors")
    require(report.get("planned_policy_steps") == FINAL_STEP and report.get("last_policy_step") == FINAL_STEP and report.get("last_logged_policy_step") == FINAL_STEP, "Inspection does not confirm exact 1M policy steps")
    require(report.get("finite_check", {}).get("all_recorded_values_finite") is True, "Inspection scalars are missing/non-finite")
    require(path_equal(report["run_dir"], row["run_dir"], base), "Inspection belongs to different run")
    require(path_equal(report["invocation_path"], row["invocation_path"], base), "Inspection belongs to different invocation")
    require(path_equal(validation["report"], row["inspection_path"], base), "Validation points to different inspection")
    require(validation.get("expected_policy_steps") == FINAL_STEP and validation.get("errors") == [], "Validation budget/errors mismatch")
    for domain in ("source", "target"):
        point = report.get("latest_{}_raw_return".format(domain))
        require(point and point.get("finite") is True and point.get("step") == FINAL_STEP, "Inspection missing final " + domain + " evaluation")


def event_metrics(run, study):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = sorted((run / "tb").glob("events.out.tfevents.*"))
    require(bool(files), "Completed run has no TensorBoard event files")
    accumulator = EventAccumulator(str(run / "tb"), size_guidance={"scalars": 0}, purge_orphaned_data=False)
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    require(all(tag in tags for tag in REQUIRED_TAGS), "Required TensorBoard tags missing: " + str(sorted(set(REQUIRED_TAGS) - set(tags))))
    scalars = {}
    counts = {}
    for tag in tags:
        values = accumulator.Scalars(tag)
        require(bool(values), "Empty scalar tag: " + tag)
        require(all(math.isfinite(float(event.value)) for event in values), "Non-finite TensorBoard value: " + tag)
        steps = [event.step for event in values]
        require(len(set(steps)) == len(steps), "Duplicate TensorBoard steps: " + tag)
        require(all(b > a for a, b in zip(steps, steps[1:])), "Regressing TensorBoard steps: " + tag)
        scalars[tag] = {event.step: float(event.value) for event in values}
        counts[tag] = len(values)
    expected = list(range(study["eval_frequency"], FINAL_STEP + 1, study["eval_frequency"]))
    for tag in ("test/source_return", "test/target_return"):
        require(sorted(scalars[tag]) == expected, "Missing/unexpected exact evaluation steps: " + tag)
    for tag in ("train/adv", "train/value", "train/q1"):
        require(max(scalars[tag]) == FINAL_STEP, "Missing exact final policy scalar: " + tag)
    minimum, maximum = study["normalization"]["ref_min"], study["normalization"]["ref_max"]
    normalize = lambda raw: 100.0 * (raw - minimum) / (maximum - minimum)
    target = scalars["test/target_return"]
    curve = [{"step": step, "raw_return": target[step], "normalized_score": normalize(target[step])} for step in expected]
    last5 = [point for point in curve if point["step"] in LAST5_STEPS]
    require(len(last5) == 5, "Missing exact Last5 endpoint")
    return {
        "final": dict(curve[-1]),
        "last5": {"steps": LAST5_STEPS, "raw_return_mean": statistics.mean(p["raw_return"] for p in last5),
                  "normalized_score_mean": statistics.mean(p["normalized_score"] for p in last5), "points": last5},
        "final_source_raw_return": scalars["test/source_return"][FINAL_STEP],
        "target_curve": curve, "scalar_counts": counts,
        "event_files": [{"path": str(path), "sha256": digest(path)} for path in files],
    }


def inspect_seed(study, row, base, context):
    result = {"seed": row["seed"], "status": "pending", "run_dir": str(resolve(row["run_dir"], base)),
              "reason": "No confirmed completion evidence", "errors": []}
    try:
        invocation_path = resolve(row["invocation_path"], base)
        invocation = read_json(invocation_path) if invocation_path.is_file() else None
        if invocation is not None:
            validate_invocation(study, row, invocation, base, context)
        completion = row["completion"]
        status_path = resolve(completion["status_path"], base)
        if not status_path.is_file():
            result["reason"] = "Completion status not created"
            return result
        status = read_json(status_path)
        stages = [stage for stage in status.get("stages", []) if stage.get("stage") == completion["train_stage"]]
        require(len(stages) <= 1, "Duplicate matching Train stages")
        validation = nested(status, completion["validation_path"])
        declared_complete = status.get("status") == "completed" or (isinstance(validation, dict) and validation.get("passed") is True)
        if not stages or not stages[0].get("ended_utc"):
            require(not declared_complete, "Completion declared without a uniquely ended Train stage")
            result["reason"] = "Train stage not started or process exit not yet confirmed"
            return result
        stage = stages[0]
        require(type(stage.get("exit_code")) is int and stage["exit_code"] == 0, "Train stage ended with nonzero/unknown exit code")
        require(invocation is not None, "Ended Train stage is missing invocation")
        if validation is None:
            require(not declared_complete, "Completed pipeline missing per-seed validation")
            result["reason"] = "Train exited; per-seed validation is still pending"
            return result
        require(isinstance(validation, dict) and validation.get("passed") is True, "Per-seed validation did not pass")
        inspection_path = resolve(row["inspection_path"], base)
        require(inspection_path.is_file(), "Validated run missing inspection report")
        report = read_json(inspection_path)
        validate_inspection(study, row, report, validation, base)
        run = resolve(row["run_dir"], base)
        require(run.is_dir(), "Validated run directory missing")
        metrics = event_metrics(run, study)
        # float32 TensorBoard values are the agreed statistical source; the
        # console can retain extra digits. Accept only float32 rounding noise.
        raw = metrics["final"]["raw_return"]
        require(math.isclose(float(report["latest_target_raw_return"]["value"]), raw, rel_tol=2e-7, abs_tol=1e-5), "Inspection final target differs from actual event data")
        result.update(metrics)
        result.update({"status": "complete", "reason": "Actual exit 0, passed validation, matching protocol, complete finite TensorBoard series",
                       "elapsed_seconds": (iso_time(stage["ended_utc"]) - iso_time(stage["started_utc"])).total_seconds(),
                       "completion_evidence": {"status_path": str(status_path), "status_sha256": digest(status_path),
                                               "train_stage": completion["train_stage"], "train_exit_code": 0,
                                               "train_ended_utc": stage["ended_utc"], "validation_path": completion["validation_path"],
                                               "inspection_path": str(inspection_path), "inspection_sha256": digest(inspection_path),
                                               "invocation_path": str(invocation_path), "invocation_sha256": digest(invocation_path)}})
    except (InvalidEvidence, OSError, KeyError, TypeError, ValueError) as exc:
        result.update({"status": "invalid", "reason": str(exc), "errors": [str(exc)]})
    return result


def aggregate(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) >= 2 else None,
            "std_ddof": 1, "unit": "target normalized score"}


def build_summary(study, study_path, validate_seed=None):
    base = study_path.parent
    context = validate_study(study, base)
    results = [inspect_seed(study, row, base, context) for row in sorted(study["seeds"], key=lambda row: row["seed"])]
    complete = [row for row in results if row["status"] == "complete"]
    invalid = [row for row in results if row["status"] == "invalid"]
    status = "invalid" if invalid else ("complete" if len(complete) == 5 else "partial")
    requested = next((row for row in results if row["seed"] == validate_seed), None) if validate_seed is not None else None
    require(validate_seed is None or validate_seed in EXPECTED_SEEDS, "Requested seed is outside study")
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "status": status,
            "task": study["task"], "study_path": str(study_path), "study_sha256": digest(study_path),
            "expected_seeds": EXPECTED_SEEDS, "completed_seeds": [row["seed"] for row in complete],
            "pending_seeds": [row["seed"] for row in results if row["status"] == "pending"],
            "invalid_seeds": [row["seed"] for row in invalid], "n_complete": len(complete), "n_expected": 5,
            "validate_seed": validate_seed, "requested_seed_passed": requested["status"] == "complete" if requested else None,
            "metrics": {"Final": aggregate([row["final"]["normalized_score"] for row in complete]),
                        "Last5": aggregate([row["last5"]["normalized_score_mean"] for row in complete])},
            "metric_definitions": study["metrics"], "normalization": study["normalization"],
            "paper_reference": study["paper_reference"], "paper_reproduction_success": None,
            "interpretation": "N<5 is partial. Final and Last5 are distinct. Paper checkpoint aggregation/SD convention is unspecified; no automatic reproduction-success verdict.",
            "selection_note": study.get("selection_note"), "environment_note": study.get("environment_note"),
            "verified_locked_files": context["checked_files"], "seeds": results,
            "errors": [{"seed": row["seed"], "message": row["reason"]} for row in invalid]}


def number(value):
    return "—" if value is None else "{:.4f}".format(value)


def markdown(summary):
    lines = ["# OTDF 五个随机种子复现汇总", "", "任务：`{}`。".format(summary.get("task", "未知")), "",
             "状态：**{}**；已核验完成 **{}/5** 个种子。".format(summary["status"], summary.get("n_complete", 0)), "",
             "## 统一统计口径", "",
             "主指标 **Final**：每个种子在恰好 1,000,000 个策略更新步时，目标域 10 个评估回合的平均原始回报，按官方 `info.py` 归一化；随后在独立种子之间计算均值和样本标准差（ddof=1）。", "",
             "辅助指标 **Last5**：先在每个种子内部平均 960,000、970,000、980,000、990,000、1,000,000 步的五个目标域归一化分数，再对种子均值计算跨种子均值和样本标准差（ddof=1）。这五次评估不能作为五个独立种子。", "",
             "数值来源为 TensorBoard 实际标签 `test/target_return` 的 float32 标量；归一化不裁剪到 0–100。只有实际训练退出码为 0、单种子检查通过、参数与固定契约一致且终点评估完整有限的运行才纳入。N<2 时标准差不计算，记为 —。", "",
             "## 每个种子的结果", "", "| Seed | 状态 | Final 原始回报 | Final 归一化分数 | Last5 归一化均值 |", "|---:|---|---:|---:|---:|"]
    for row in summary.get("seeds", []):
        lines.append("| {} | {} | {} | {} | {} |".format(row["seed"], row["status"], number(row.get("final", {}).get("raw_return")), number(row.get("final", {}).get("normalized_score")), number(row.get("last5", {}).get("normalized_score_mean"))))
    lines += ["", "## 主指标 Final", ""]
    final = summary.get("metrics", {}).get("Final", {})
    lines += ["N={}；均值={}；样本标准差={}（ddof=1）。".format(final.get("n", 0), number(final.get("mean")), number(final.get("sample_sd"))), "", "## 辅助指标 Last5", ""]
    last5 = summary.get("metrics", {}).get("Last5", {})
    lines += ["N={}；均值={}；样本标准差={}（ddof=1）。".format(last5.get("n", 0), number(last5.get("mean")), number(last5.get("sample_sd"))), "", "## 与论文比较时的边界", "",
              "论文表 1 的对应结果为 **50.5 ± 5.8（5 个随机种子）**。论文的检查点聚合方式以及标准差 ddof 未明确，不能直接认定与这里的 Final 或 Last5 完全同口径，也不能由一次单种子结果或数值接近自动判定复现成功。", "",
              "不足 5 个已完成种子时，本表只是一份 partial 中间汇总。若状态为 invalid，存在未解决的数据或证据冲突，结果仅供排查。"]
    if summary.get("selection_note"):
        lines += ["", "运行选择说明：" + summary["selection_note"]]
    if summary.get("environment_note"):
        lines += ["", "环境说明：" + summary["environment_note"]]
    incomplete = [row for row in summary.get("seeds", []) if row["status"] != "complete"]
    if incomplete or summary.get("errors"):
        lines += ["", "## 未纳入的种子及原因", ""]
        lines.extend("- Seed {}（{}）：{}".format(row["seed"], row["status"], row["reason"]) for row in incomplete)
        if not incomplete:
            lines.extend("- " + str(error) for error in summary.get("errors", []))
    lines += ["", "固定契约 SHA256：`{}`。".format(summary.get("study_sha256", "不可用")), "",
              "生成时间（UTC）：{}。原始日志、模型、数据与契约未被修改。".format(summary["created_utc"]), ""]
    return "\n".join(lines)


def csv_text(summary):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["seed", "status", "included", "final_step", "final_target_raw_return", "final_target_normalized_score", "last5_target_normalized_mean", "elapsed_seconds", "run_dir", "reason"])
    for row in summary.get("seeds", []):
        writer.writerow([row["seed"], row["status"], row["status"] == "complete", row.get("final", {}).get("step", ""),
                         row.get("final", {}).get("raw_return", ""), row.get("final", {}).get("normalized_score", ""),
                         row.get("last5", {}).get("normalized_score_mean", ""), row.get("elapsed_seconds", ""), row["run_dir"], row["reason"]])
    return output.getvalue()


def atomic_write(path, content, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline="", dir=str(path.parent), prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, str(path))
        name = None
    finally:
        if name is not None:
            Path(name).unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-seed", type=int)
    args = parser.parse_args()
    study_path = args.study.resolve()
    try:
        study = read_json(study_path)
        summary = build_summary(study, study_path, args.validate_seed)
        code = 1 if summary["status"] == "invalid" or summary["requested_seed_passed"] is False else 0
    except (InvalidEvidence, OSError, KeyError, TypeError, ValueError) as exc:
        summary = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "invalid",
                   "n_complete": 0, "n_expected": 5, "study_path": str(study_path), "errors": [str(exc)], "seeds": []}
        code = 1
    except Exception as exc:
        print(json.dumps({"status": "invalid", "errors": ["{}: {}".format(type(exc).__name__, exc)]}, ensure_ascii=False))
        return 2
    try:
        output = args.output.resolve()
        atomic_write(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        atomic_write(output / "per_seed.csv", csv_text(summary), encoding="utf-8-sig")
        atomic_write(output / "SUMMARY.md", markdown(summary))
    except Exception as exc:
        print(json.dumps({"status": "invalid", "errors": ["Output failed: {}: {}".format(type(exc).__name__, exc)]}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": summary["status"], "n_complete": summary["n_complete"], "n_expected": 5,
                      "metrics": summary.get("metrics"), "requested_seed_passed": summary.get("requested_seed_passed"),
                      "errors": summary["errors"], "output": str(output)}, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
