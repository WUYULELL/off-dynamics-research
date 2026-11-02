import os
import shutil
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUT_DIR = Path("analysis_results/priority_walker2d_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TB_CACHE_DIR = OUT_DIR / ".tb_event_cache"
TB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENV_NAME = "walker2d-kinematic-footjnt"
DESCRIPTOR = f"{ENV_NAME}-srcdatatype-medium-tardatatype-medium-medium"

RUNS = [
    {"label": "IQL-target", "policy": "IQL", "root": Path("runs_paper_like_iql_target_only"), "source_weight": False},
    {"label": "IQL-mix", "policy": "IQL", "root": Path("runs_paper_like_iql_unc"), "source_weight": False},
    {"label": "IQL_CONST-w033", "policy": "IQL_UNC", "root": Path("runs_walker2d_iql_const_w033"), "source_weight": True},
    {"label": "IQL_UNC-a1", "policy": "IQL_UNC", "root": Path("runs_paper_like_iql_unc"), "source_weight": True},
    {"label": "IQL_UNC-a2", "policy": "IQL_UNC", "root": Path("runs_iql_unc_alpha_ablation/alpha_2p0"), "source_weight": True},
    {"label": "TD3_BC-target", "policy": "TD3_BC", "root": Path("runs_paper_like_td3_bc_target_only"), "source_weight": False},
    {"label": "TD3_BC-mix", "policy": "TD3_BC", "root": Path("runs_paper_like_td3_bc_mix"), "source_weight": False},
    {"label": "BOSA", "policy": "BOSA", "root": Path("runs_paper_like_sweep"), "source_weight": False},
    {"label": "IGDF", "policy": "IGDF", "root": Path("runs_paper_like_sweep"), "source_weight": False},
]

TAGS = [
    "test/target_normalized_score",
    "test/target_return",
    "train/source_weight_mean",
    "train/source_uncertainty_mean",
]


def read_tb(tb_dir: Path, run: Dict, seed: int) -> List[Dict]:
    files = sorted(tb_dir.glob("events.out.tfevents*"))
    if not files:
        return []
    event_file = files[-1]
    event_path = str(event_file.resolve())
    if os.name == "nt" and len(event_path) >= 240:
        cached_file = TB_CACHE_DIR / f"{run['label'].replace('/', '_')}_seed{seed}.tfevents"
        long_event_path = event_path if event_path.startswith("\\\\?\\") else f"\\\\?\\{event_path}"
        shutil.copyfile(long_event_path, cached_file)
        event_path = str(cached_file.resolve())
    acc = EventAccumulator(event_path)
    acc.Reload()
    scalar_tags = acc.Tags().get("scalars", [])
    rows = []
    for tag in TAGS:
        if tag not in scalar_tags:
            continue
        for event in acc.Scalars(tag):
            rows.append(
                {
                    "label": run["label"],
                    "policy": run["policy"],
                    "seed": seed,
                    "tag": tag,
                    "step": event.step,
                    "value": event.value,
                }
            )
    return rows


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


rows = []
missing = []
for run in RUNS:
    for seed in range(5):
        tb_dir = run["root"] / run["policy"] / DESCRIPTOR / f"r{seed}" / "tb"
        run_rows = read_tb(tb_dir, run, seed)
        if not run_rows:
            missing.append({"label": run["label"], "seed": seed, "tb_dir": str(tb_dir)})
        rows.extend(run_rows)

pd.DataFrame(missing).to_csv(OUT_DIR / "missing_runs.csv", index=False)

if not rows:
    raise SystemExit("No TensorBoard events found.")

scalars = pd.DataFrame(rows)
scalars.to_csv(OUT_DIR / "priority_walker2d_scalars.csv", index=False)

summary_rows = []
for keys, df in scalars.groupby(["label", "policy", "seed", "tag"]):
    label, policy, seed, tag = keys
    df = df.sort_values("step")
    values = df["value"]
    best_idx = values.idxmax()
    last5 = values.tail(5)
    summary_rows.append(
        {
            "label": label,
            "policy": policy,
            "seed": seed,
            "metric": tag,
            "initial": values.iloc[0],
            "final": values.iloc[-1],
            "best": df.loc[best_idx, "value"],
            "best_step": int(df.loc[best_idx, "step"]),
            "last5_avg": last5.mean(),
            "last5_std": last5.std(ddof=1) if len(last5) > 1 else 0.0,
            "mean": values.mean(),
            "min": values.min(),
            "max": values.max(),
        }
    )

by_seed = pd.DataFrame(summary_rows)
by_seed.to_csv(OUT_DIR / "priority_walker2d_summary_by_seed.csv", index=False)

score_seed = by_seed[by_seed["metric"] == "test/target_normalized_score"].copy()
score_summary = (
    score_seed.groupby(["label", "policy"])
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
        weight_seed.groupby(["label"])
        .agg(source_weight_mean=("mean", "mean"), source_weight_std=("mean", "std"))
        .reset_index()
    )
    score_summary = score_summary.merge(weight_summary, on="label", how="left")
else:
    score_summary["source_weight_mean"] = pd.NA
    score_summary["source_weight_std"] = pd.NA

order = [run["label"] for run in RUNS]
score_summary["label"] = pd.Categorical(score_summary["label"], categories=order, ordered=True)
score_summary = score_summary.sort_values("label")
score_summary.to_csv(OUT_DIR / "priority_walker2d_score_summary.csv", index=False)

# Paper-ready table with exact mean +/- std values.
paper_table = score_summary[
    ["label", "seeds", "final_mean", "final_std", "best_mean", "best_std", "last5_mean", "last5_std"]
].copy()
paper_table["Final"] = paper_table.apply(
    lambda row: f"{row['final_mean']:.2f} ± {row['final_std']:.2f}", axis=1
)
paper_table["Best"] = paper_table.apply(
    lambda row: f"{row['best_mean']:.2f} ± {row['best_std']:.2f}", axis=1
)
paper_table["Last-5 Avg"] = paper_table.apply(
    lambda row: f"{row['last5_mean']:.2f} ± {row['last5_std']:.2f}", axis=1
)
paper_table = paper_table.rename(columns={"label": "Method", "seeds": "Seeds"})[
    ["Method", "Seeds", "Final", "Best", "Last-5 Avg"]
]
paper_table.to_csv(OUT_DIR / "priority_walker2d_paper_table.csv", index=False, encoding="utf-8-sig")

paired = score_seed.pivot_table(
    index="seed",
    columns="label",
    values=["final", "best", "last5_avg"],
    aggfunc="first",
)
paired.columns = [f"{metric}_{label}" for metric, label in paired.columns]
paired = paired.reset_index()
paired.to_csv(OUT_DIR / "priority_walker2d_paired_by_seed.csv", index=False)

# Compact paired-seed table for the primary comparison requested in the report.
primary_methods = ["IQL-mix", "IQL_CONST-w033", "IQL_UNC-a2", "IGDF"]
primary_columns = ["seed"] + [
    f"last5_avg_{method}" for method in primary_methods if f"last5_avg_{method}" in paired.columns
]
primary_paired = paired[primary_columns].copy()
primary_paired = primary_paired.rename(
    columns={
        "seed": "Seed",
        "last5_avg_IQL-mix": "IQL-mix",
        "last5_avg_IQL_CONST-w033": "IQL_CONST-w033",
        "last5_avg_IQL_UNC-a2": "IQL_UNC-a2",
        "last5_avg_IGDF": "IGDF",
    }
)
for column in primary_paired.columns[1:]:
    primary_paired[column] = primary_paired[column].round(2)
primary_paired.to_csv(
    OUT_DIR / "priority_walker2d_primary_paired_last5.csv", index=False, encoding="utf-8-sig"
)

win_rows = []
baseline = "IQL-mix"
for method in ["IQL-target", "IQL_CONST-w033", "IQL_UNC-a1", "IQL_UNC-a2", "TD3_BC-target", "TD3_BC-mix", "BOSA", "IGDF"]:
    for metric in ["final", "best", "last5_avg"]:
        a = f"{metric}_{method}"
        b = f"{metric}_{baseline}"
        if a in paired.columns and b in paired.columns:
            valid = paired[[a, b]].dropna()
            if not valid.empty:
                win_rows.append(
                    {
                        "method": method,
                        "baseline": baseline,
                        "metric": metric,
                        "wins": int((valid[a] > valid[b]).sum()),
                        "ties": int((valid[a] == valid[b]).sum()),
                        "losses": int((valid[a] < valid[b]).sum()),
                        "mean_delta": (valid[a] - valid[b]).mean(),
                    }
                )
pd.DataFrame(win_rows).to_csv(OUT_DIR / "priority_walker2d_paired_wins_vs_iql_mix.csv", index=False)

primary_win_rows = []
for method in ["IQL_CONST-w033", "IQL_UNC-a2", "IGDF"]:
    for metric in ["final", "best", "last5_avg"]:
        method_col = f"{metric}_{method}"
        baseline_col = f"{metric}_{baseline}"
        if method_col not in paired.columns or baseline_col not in paired.columns:
            continue
        valid = paired[["seed", method_col, baseline_col]].dropna()
        deltas = valid[method_col] - valid[baseline_col]
        primary_win_rows.append(
            {
                "method": method,
                "baseline": baseline,
                "metric": metric,
                "wins": int((deltas > 0).sum()),
                "ties": int((deltas == 0).sum()),
                "losses": int((deltas < 0).sum()),
                "mean_delta": deltas.mean(),
                "median_delta": deltas.median(),
            }
        )
pd.DataFrame(primary_win_rows).to_csv(
    OUT_DIR / "priority_walker2d_primary_paired_summary.csv", index=False
)

plot_score = score_summary.dropna(subset=["final_mean"]).copy()
fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=180)
for ax, col, title in [
    (axes[0], "final_mean", "Final"),
    (axes[1], "best_mean", "Best"),
    (axes[2], "last5_mean", "Last-5 Avg"),
]:
    err_col = col.replace("_mean", "_std")
    ax.bar(plot_score["label"].astype(str), plot_score[col], yerr=plot_score[err_col], capsize=3)
    ax.set_title(title)
    ax.set_ylabel("Target normalized score")
    ax.tick_params(axis="x", rotation=35)
fig.suptitle("Walker2d Paper-Like 10% Target | Priority Comparison")
plt.tight_layout()
plt.savefig(OUT_DIR / "priority_walker2d_scores.png", bbox_inches="tight")
plt.close()

# Plot the uncertainty mechanism over training for alpha=1 and alpha=2.
mechanism = scalars[
    scalars["label"].isin(["IQL_UNC-a1", "IQL_UNC-a2"])
    & scalars["tag"].isin(
        [
            "train/source_weight_mean",
            "train/source_uncertainty_mean",
            "test/target_normalized_score",
        ]
    )
].copy()
if not mechanism.empty:
    mechanism_summary = (
        mechanism.groupby(["label", "tag", "step"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    mechanism_summary["std"] = mechanism_summary["std"].fillna(0.0)
    mechanism_summary.to_csv(OUT_DIR / "priority_walker2d_unc_mechanism_curves.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=180)
    mechanism_specs = [
        ("train/source_weight_mean", "Source weight mean", "Weight"),
        ("train/source_uncertainty_mean", "Source uncertainty mean", "Uncertainty"),
        ("test/target_normalized_score", "Target normalized score", "Score"),
    ]
    colors = {"IQL_UNC-a1": "#2878B5", "IQL_UNC-a2": "#D95319"}
    for ax, (tag, title, ylabel) in zip(axes, mechanism_specs):
        for label in ["IQL_UNC-a1", "IQL_UNC-a2"]:
            df = mechanism_summary[
                (mechanism_summary["label"] == label) & (mechanism_summary["tag"] == tag)
            ].sort_values("step")
            if df.empty:
                continue
            x = df["step"].to_numpy()
            mean = df["mean"].to_numpy()
            std = df["std"].to_numpy()
            ax.plot(x, mean, label=label, color=colors[label], linewidth=1.8)
            ax.fill_between(x, mean - std, mean + std, color=colors[label], alpha=0.16)
        ax.set_title(title)
        ax.set_xlabel("Gradient steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Walker2d IQL_UNC Mechanism | Mean ± Std Across 5 Seeds")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "priority_walker2d_unc_mechanism.png", bbox_inches="tight")
    plt.close()

    aligned_rows = []
    for label in ["IQL_UNC-a1", "IQL_UNC-a2"]:
        for seed in range(5):
            seed_df = mechanism[
                (mechanism["label"] == label) & (mechanism["seed"] == seed)
            ]
            score_df = seed_df[seed_df["tag"] == "test/target_normalized_score"][
                ["step", "value"]
            ].rename(columns={"value": "target_score"})
            weight_df = seed_df[seed_df["tag"] == "train/source_weight_mean"][
                ["step", "value"]
            ].rename(columns={"value": "source_weight"})
            uncertainty_df = seed_df[seed_df["tag"] == "train/source_uncertainty_mean"][
                ["step", "value"]
            ].rename(columns={"value": "source_uncertainty"})
            aligned = score_df.merge(weight_df, on="step", how="inner").merge(
                uncertainty_df, on="step", how="inner"
            )
            if aligned.empty:
                continue
            aligned["label"] = label
            aligned["seed"] = seed
            aligned_rows.append(aligned)

    if aligned_rows:
        aligned_metrics = pd.concat(aligned_rows, ignore_index=True)
        aligned_metrics.to_csv(
            OUT_DIR / "priority_walker2d_score_weight_aligned.csv", index=False
        )
        correlation_rows = []
        for (label, seed), df in aligned_metrics.groupby(["label", "seed"]):
            correlation_rows.append(
                {
                    "label": label,
                    "seed": seed,
                    "score_weight_corr": df["target_score"].corr(df["source_weight"]),
                    "score_uncertainty_corr": df["target_score"].corr(
                        df["source_uncertainty"]
                    ),
                    "points": len(df),
                }
            )
        correlation_df = pd.DataFrame(correlation_rows)
        correlation_df.to_csv(
            OUT_DIR / "priority_walker2d_score_weight_correlations.csv", index=False
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
        for label in ["IQL_UNC-a1", "IQL_UNC-a2"]:
            df = aligned_metrics[aligned_metrics["label"] == label]
            axes[0].scatter(
                df["source_weight"],
                df["target_score"],
                s=18,
                alpha=0.45,
                label=label,
                color=colors[label],
            )
            axes[1].scatter(
                df["source_uncertainty"],
                df["target_score"],
                s=18,
                alpha=0.45,
                label=label,
                color=colors[label],
            )
        axes[0].set_xlabel("Source weight mean")
        axes[0].set_ylabel("Target normalized score")
        axes[0].set_title("Score vs source weight")
        axes[1].set_xlabel("Source uncertainty mean")
        axes[1].set_ylabel("Target normalized score")
        axes[1].set_title("Score vs source uncertainty")
        for ax in axes:
            ax.grid(alpha=0.2)
        axes[0].legend(frameon=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "priority_walker2d_score_weight_scatter.png", bbox_inches="tight")
        plt.close()

report_lines = [
    "# Walker2d Paper-Like 10% Target Results",
    "",
    "## Exact Results",
    "",
    dataframe_to_markdown(paper_table),
    "",
    "## Primary Paired Seed Comparison (Last-5 Average)",
    "",
    dataframe_to_markdown(primary_paired),
    "",
    "## Interpretation",
    "",
    "- IQL-mix and IQL-target have nearly identical Last-5 performance, so naive source-data mixing gives no clear benefit.",
    "- IQL_UNC-a2 has the highest Last-5 mean and the lowest Last-5 standard deviation among the compared methods.",
    "- IQL_UNC-a1 does not improve over IQL-mix, indicating that stronger uncertainty suppression is important in this task.",
    "- TD3_BC reaches high best scores but is unstable late in training; Best alone is therefore misleading.",
    "- These conclusions currently apply to Walker2d only. Hopper and HalfCheetah are still required for cross-environment evidence.",
]
(OUT_DIR / "WALKER2D_RESULTS_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(score_summary.to_string(index=False))
if missing:
    print("\nMissing runs:")
    print(pd.DataFrame(missing).to_string(index=False))
