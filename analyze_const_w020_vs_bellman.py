import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUT_DIR = Path("analysis_results/const_w020_vs_bellman")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TB_CACHE_DIR = OUT_DIR / ".tb_event_cache"
TB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENVS = [
    "hopper-kinematic-legjnt",
    "walker2d-kinematic-footjnt",
    "halfcheetah-kinematic-footjnt",
]

ENV_SHORT = {
    "hopper-kinematic-legjnt": "Hopper",
    "walker2d-kinematic-footjnt": "Walker2d",
    "halfcheetah-kinematic-footjnt": "HalfCheetah",
}

METHODS = [
    "IQL-target",
    "IQL-mix",
    "IQL_CONST-w0.20",
    "IQL_BELLMAN_FB-b0.20",
    "IGDF",
]

RUNS = {
    "IQL-target": {
        "policy": "IQL",
        "roots": {
            "hopper-kinematic-legjnt": Path("runs_cross_env_iql_target"),
            "walker2d-kinematic-footjnt": Path("runs_paper_like_iql_target_only"),
            "halfcheetah-kinematic-footjnt": Path("runs_cross_env_iql_target"),
        },
    },
    "IQL-mix": {
        "policy": "IQL",
        "roots": {
            "hopper-kinematic-legjnt": Path("runs_cross_env_iql_mix"),
            "walker2d-kinematic-footjnt": Path("runs_paper_like_iql_unc"),
            "halfcheetah-kinematic-footjnt": Path("runs_cross_env_iql_mix"),
        },
    },
    "IQL_CONST-w0.20": {
        "policy": "IQL_UNC",
        "roots": {env: Path("runs_iql_const_w020_3env") for env in ENVS},
    },
    "IQL_BELLMAN_FB-b0.20": {
        "policy": "IQL_BELLMAN_FB",
        "roots": {
            "hopper-kinematic-legjnt": Path("runs_hopper_bellman_b020_5seeds/b0p20"),
            "walker2d-kinematic-footjnt": Path("runs_walker2d_bellman_budget_sweep/b0p20"),
            "halfcheetah-kinematic-footjnt": Path("runs_budget_ablation_bellman_fb/b0p20"),
        },
    },
    "IGDF": {
        "policy": "IGDF",
        "roots": {
            "hopper-kinematic-legjnt": Path("runs_cross_env_igdf"),
            "walker2d-kinematic-footjnt": Path("runs_paper_like_sweep"),
            "halfcheetah-kinematic-footjnt": Path("runs_cross_env_igdf"),
        },
    },
}

TAGS = [
    "test/target_normalized_score",
    "train/source_weight_mean",
    "train/source_weight_min",
    "train/source_weight_max",
    "train/source_weight_ess_ratio",
    "train/bellman_mismatch_mean",
    "train/corr_bellman_mismatch_source_weight",
]


def event_path_for_reader(event_file: Path, cache_name: str) -> str:
    event_path = str(event_file.resolve())
    if os.name != "nt" or len(event_path) < 240:
        return event_path

    cached_file = TB_CACHE_DIR / cache_name
    long_path = event_path if event_path.startswith("\\\\?\\") else f"\\\\?\\{event_path}"
    shutil.copyfile(long_path, cached_file)
    return str(cached_file.resolve())


def read_tb(tb_dir: Path, label: str, policy: str, env: str, seed: int):
    files = sorted(tb_dir.glob("events.out.tfevents*"))
    if not files:
        return [], "no_event_file"

    candidates = []
    errors = []
    safe_label = label.replace("/", "_").replace(".", "p")
    for index, event_file in enumerate(files):
        try:
            cache_name = f"{ENV_SHORT[env]}_{safe_label}_seed{seed}_{index}.tfevents"
            event_path = event_path_for_reader(event_file, cache_name)
            acc = EventAccumulator(event_path, size_guidance={"scalars": 0})
            acc.Reload()
            scalar_tags = acc.Tags().get("scalars", [])
            if "test/target_normalized_score" not in scalar_tags:
                errors.append(f"{event_file.name}: missing score tag")
                continue
            score_events = acc.Scalars("test/target_normalized_score")
            if not score_events:
                errors.append(f"{event_file.name}: empty score tag")
                continue
            candidates.append(
                (
                    int(score_events[-1].step),
                    len(score_events),
                    event_file.stat().st_mtime,
                    acc,
                    scalar_tags,
                    event_file,
                )
            )
        except Exception as exc:
            errors.append(f"{event_file.name}: {type(exc).__name__}: {exc}")

    if not candidates:
        return [], "; ".join(errors) if errors else "no_readable_event_file"

    _, _, _, acc, scalar_tags, chosen_file = max(candidates, key=lambda item: item[:3])
    rows = []
    for tag in TAGS:
        if tag not in scalar_tags:
            continue
        for event in acc.Scalars(tag):
            rows.append(
                {
                    "env": env,
                    "env_short": ENV_SHORT[env],
                    "label": label,
                    "policy": policy,
                    "seed": seed,
                    "tag": tag,
                    "step": int(event.step),
                    "value": float(event.value),
                    "event_file": str(chosen_file),
                }
            )
    return rows, ""


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display = df.fillna("")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def bootstrap_mean_ci(values, samples=20000, seed=20260917):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


rows = []
missing = []
for env in ENVS:
    descriptor = f"{env}-srcdatatype-medium-tardatatype-medium-medium"
    for label in METHODS:
        run = RUNS[label]
        policy = run["policy"]
        root = run["roots"][env]
        for seed in range(5):
            tb_dir = root / policy / descriptor / f"r{seed}" / "tb"
            run_rows, error = read_tb(tb_dir, label, policy, env, seed)
            if not run_rows:
                missing.append(
                    {
                        "env": env,
                        "label": label,
                        "seed": seed,
                        "tb_dir": str(tb_dir),
                        "reason": error,
                    }
                )
            rows.extend(run_rows)

pd.DataFrame(missing).to_csv(OUT_DIR / "missing_runs.csv", index=False, encoding="utf-8-sig")
if not rows:
    raise SystemExit("No readable TensorBoard events found.")

scalars = pd.DataFrame(rows)
scalars.to_csv(OUT_DIR / "scalars.csv", index=False, encoding="utf-8-sig")

summary_rows = []
for keys, frame in scalars.groupby(["env", "env_short", "label", "policy", "seed", "tag"]):
    env, env_short, label, policy, seed, tag = keys
    frame = frame.sort_values("step")
    values = frame["value"]
    best_index = values.idxmax()
    summary_rows.append(
        {
            "env": env,
            "env_short": env_short,
            "label": label,
            "policy": policy,
            "seed": int(seed),
            "metric": tag,
            "evaluations": int(len(values)),
            "final_step": int(frame["step"].iloc[-1]),
            "final": float(values.iloc[-1]),
            "best": float(frame.loc[best_index, "value"]),
            "best_step": int(frame.loc[best_index, "step"]),
            "last5_avg": float(values.tail(5).mean()),
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    )

by_seed = pd.DataFrame(summary_rows)
by_seed.to_csv(OUT_DIR / "summary_by_seed_all_metrics.csv", index=False, encoding="utf-8-sig")

score_seed = by_seed[by_seed["metric"] == "test/target_normalized_score"].copy()
score_seed.to_csv(OUT_DIR / "score_by_seed.csv", index=False, encoding="utf-8-sig")

score_summary = (
    score_seed.groupby(["env", "env_short", "label", "policy"])
    .agg(
        seeds=("seed", "nunique"),
        final_mean=("final", "mean"),
        final_std=("final", "std"),
        best_mean=("best", "mean"),
        best_std=("best", "std"),
        last5_mean=("last5_avg", "mean"),
        last5_std=("last5_avg", "std"),
    )
    .reset_index()
)

weight_seed = by_seed[by_seed["metric"] == "train/source_weight_mean"].copy()
if not weight_seed.empty:
    weight_summary = (
        weight_seed.groupby(["env", "label"])
        .agg(source_weight_mean=("mean", "mean"), source_weight_std=("mean", "std"))
        .reset_index()
    )
    score_summary = score_summary.merge(weight_summary, on=["env", "label"], how="left")
else:
    score_summary["source_weight_mean"] = np.nan
    score_summary["source_weight_std"] = np.nan

env_order = {env: index for index, env in enumerate(ENVS)}
method_order = {method: index for index, method in enumerate(METHODS)}
score_summary["env_order"] = score_summary["env"].map(env_order)
score_summary["method_order"] = score_summary["label"].map(method_order)
score_summary = score_summary.sort_values(["env_order", "method_order"]).drop(
    columns=["env_order", "method_order"]
)
score_summary.to_csv(OUT_DIR / "score_summary.csv", index=False, encoding="utf-8-sig")

paper_table = score_summary.copy()
paper_table["Final"] = paper_table.apply(
    lambda row: f"{row.final_mean:.2f} +/- {row.final_std:.2f}", axis=1
)
paper_table["Best"] = paper_table.apply(
    lambda row: f"{row.best_mean:.2f} +/- {row.best_std:.2f}", axis=1
)
paper_table["Last-5 Avg"] = paper_table.apply(
    lambda row: f"{row.last5_mean:.2f} +/- {row.last5_std:.2f}", axis=1
)
paper_table["Source Weight"] = paper_table.apply(
    lambda row: ""
    if pd.isna(row.source_weight_mean)
    else f"{row.source_weight_mean:.3f} +/- {row.source_weight_std:.3f}",
    axis=1,
)
paper_table = paper_table.rename(
    columns={"env_short": "Environment", "label": "Method", "seeds": "Seeds"}
)[["Environment", "Method", "Seeds", "Final", "Best", "Last-5 Avg", "Source Weight"]]
paper_table.to_csv(OUT_DIR / "paper_table.csv", index=False, encoding="utf-8-sig")

paired_frames = []
for env, frame in score_seed.groupby("env"):
    paired = frame.pivot_table(index="seed", columns="label", values="last5_avg", aggfunc="first")
    paired = paired.reindex(columns=METHODS).reset_index()
    paired.insert(0, "env_short", ENV_SHORT[env])
    paired.insert(0, "env", env)
    paired_frames.append(paired)
paired_last5 = pd.concat(paired_frames, ignore_index=True)
paired_last5.to_csv(OUT_DIR / "paired_last5.csv", index=False, encoding="utf-8-sig")

comparisons = [
    ("IQL_BELLMAN_FB-b0.20", "IQL_CONST-w0.20"),
    ("IQL_BELLMAN_FB-b0.20", "IQL-mix"),
    ("IQL_BELLMAN_FB-b0.20", "IGDF"),
    ("IQL_BELLMAN_FB-b0.20", "IQL-target"),
]
comparison_rows = []
groups = [(ENV_SHORT[env], paired_last5[paired_last5["env"] == env]) for env in ENVS]
groups.append(("All environments", paired_last5))
for group_name, paired in groups:
    for method, baseline in comparisons:
        if method not in paired.columns or baseline not in paired.columns:
            continue
        valid = paired[[method, baseline]].dropna()
        delta = valid[method] - valid[baseline]
        ci_low, ci_high = bootstrap_mean_ci(delta.to_numpy())
        comparison_rows.append(
            {
                "group": group_name,
                "method": method,
                "baseline": baseline,
                "pairs": int(len(delta)),
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
                "mean_delta": float(delta.mean()) if len(delta) else np.nan,
                "median_delta": float(delta.median()) if len(delta) else np.nan,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
        )
paired_comparison = pd.DataFrame(comparison_rows)
paired_comparison.to_csv(OUT_DIR / "paired_comparison.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
for ax, env in zip(axes, ENVS):
    frame = score_summary[score_summary["env"] == env].copy()
    frame["method_order"] = frame["label"].map(method_order)
    frame = frame.sort_values("method_order")
    x = np.arange(len(frame))
    colors = ["#4c78a8", "#72b7b2", "#f2cf5b", "#e45756", "#54a24b"][: len(frame)]
    ax.bar(x, frame["last5_mean"], yerr=frame["last5_std"], capsize=4, color=colors)
    ax.set_title(ENV_SHORT[env])
    ax.set_xticks(x)
    ax.set_xticklabels(frame["label"], rotation=28, ha="right")
    ax.set_ylabel("Target normalized score (Last-5 Avg)")
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Paper-Like 10% Target: Constant w0.20 vs Bellman b0.20")
fig.tight_layout()
fig.savefig(OUT_DIR / "last5_comparison.png", dpi=220, bbox_inches="tight")
plt.close(fig)

if {"IQL_BELLMAN_FB-b0.20", "IQL_CONST-w0.20"}.issubset(paired_last5.columns):
    delta_frame = paired_last5.dropna(
        subset=["IQL_BELLMAN_FB-b0.20", "IQL_CONST-w0.20"]
    ).copy()
    if not delta_frame.empty:
        delta_frame["delta"] = (
            delta_frame["IQL_BELLMAN_FB-b0.20"] - delta_frame["IQL_CONST-w0.20"]
        )
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        for index, env in enumerate(ENVS):
            frame = delta_frame[delta_frame["env"] == env]
            offsets = index + np.linspace(-0.18, 0.18, max(len(frame), 1))
            ax.scatter(offsets, frame["delta"], s=55, label=ENV_SHORT[env])
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(range(len(ENVS)))
        ax.set_xticklabels([ENV_SHORT[env] for env in ENVS])
        ax.set_ylabel("Last-5 delta: Bellman b0.20 - Constant w0.20")
        ax.set_title("Paired Seed Comparison at Equal Source Budget")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "bellman_minus_constant_paired.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

const_comparison = paired_comparison[
    (paired_comparison["method"] == "IQL_BELLMAN_FB-b0.20")
    & (paired_comparison["baseline"] == "IQL_CONST-w0.20")
].copy()
const_comparison_valid = const_comparison[const_comparison["pairs"] > 0].copy()
summary_lines = [
    "# Constant w0.20 vs Bellman b0.20",
    "",
    "Protocol: paper-like offline-offline, 10% target data, 500k gradient steps, seeds 0-4.",
    "Primary metric: mean target normalized score over the last five evaluations.",
    "",
    "## Exact results",
    "",
    dataframe_to_markdown(paper_table),
    "",
    "## Paired comparison",
    "",
]
if const_comparison_valid.empty:
    summary_lines.append(
        "IQL_CONST-w0.20 runs are incomplete. Run run_iql_const_w020_3env_5seeds.ps1 and rerun this analyzer."
    )
else:
    display_comparison = const_comparison_valid[
        [
            "group",
            "pairs",
            "wins",
            "ties",
            "losses",
            "mean_delta",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ]
    ].copy()
    for column in ["mean_delta", "bootstrap_ci_low", "bootstrap_ci_high"]:
        display_comparison[column] = display_comparison[column].map(lambda value: f"{value:.2f}")
    summary_lines.append(dataframe_to_markdown(display_comparison))
    summary_lines.extend(
        [
            "",
            "Positive delta means Bellman sample-level weighting outperformed uniform constant weighting at the same source budget.",
        ]
    )

if missing:
    summary_lines.extend(
        [
            "",
            "## Missing runs",
            "",
            f"{len(missing)} method-environment-seed runs were missing or unreadable; see missing_runs.csv.",
        ]
    )

(OUT_DIR / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(paper_table.to_string(index=False))
print("\nBellman b0.20 vs constant w0.20, paired Last-5:")
if const_comparison_valid.empty:
    print("Constant w0.20 runs are not complete yet.")
else:
    print(const_comparison_valid.to_string(index=False))
if missing:
    print(f"\nMissing/unreadable runs: {len(missing)} (see missing_runs.csv)")
