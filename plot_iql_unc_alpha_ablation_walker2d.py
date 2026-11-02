import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUT_DIR = Path("analysis_results/iql_unc_alpha_ablation_walker2d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENV_NAME = "walker2d-kinematic-footjnt"
RUNS = [
    {"label": "IQL", "alpha": None, "root": Path("runs_paper_like_iql_unc"), "policy": "IQL"},
    {"label": "alpha=0.1", "alpha": 0.1, "root": Path("runs_iql_unc_alpha_ablation/alpha_0p1"), "policy": "IQL_UNC"},
    {"label": "alpha=0.5", "alpha": 0.5, "root": Path("runs_iql_unc_alpha_ablation/alpha_0p5"), "policy": "IQL_UNC"},
    {"label": "alpha=1.0", "alpha": 1.0, "root": Path("runs_paper_like_iql_unc"), "policy": "IQL_UNC"},
    {"label": "alpha=2.0", "alpha": 2.0, "root": Path("runs_iql_unc_alpha_ablation/alpha_2p0"), "policy": "IQL_UNC"},
]

TAGS = [
    "test/target_normalized_score",
    "test/target_return",
    "train/source_uncertainty_mean",
    "train/source_weight_mean",
    "train/source_weight_min",
    "train/source_weight_max",
]


def load_scalars(tb_dir: Path, run: Dict, seed: int) -> List[Dict]:
    files = sorted(tb_dir.glob("events.out.tfevents*"))
    if not files:
        return []
    acc = EventAccumulator(str(files[-1]))
    acc.Reload()
    scalar_tags = acc.Tags().get("scalars", [])
    rows = []
    for tag in TAGS:
        if tag not in scalar_tags:
            continue
        rows.extend(
            {
                "label": run["label"],
                "alpha": run["alpha"],
                "policy": run["policy"],
                "seed": seed,
                "tag": tag,
                "step": event.step,
                "value": event.value,
            }
            for event in acc.Scalars(tag)
        )
    return rows


rows = []
for run in RUNS:
    descriptor = f"{ENV_NAME}-srcdatatype-medium-tardatatype-medium-medium"
    for seed in range(5):
        tb_dir = run["root"] / run["policy"] / descriptor / f"r{seed}" / "tb"
        rows.extend(load_scalars(tb_dir, run, seed))

if not rows:
    raise SystemExit("No alpha-ablation TensorBoard events found.")

scalars = pd.DataFrame(rows)
scalars.to_csv(OUT_DIR / "iql_unc_alpha_ablation_scalars.csv", index=False)

summary_rows = []
for keys, df in scalars.groupby(["label", "policy", "seed", "tag"], dropna=False):
    label, policy, seed, tag = keys
    df = df.sort_values("step")
    values = df["value"]
    best_idx = values.idxmax()
    summary_rows.append(
        {
            "label": label,
            "policy": policy,
            "seed": seed,
            "metric": tag,
            "initial": values.iloc[0],
            "final": values.iloc[-1],
            "mean": values.mean(),
            "min": values.min(),
            "max": values.max(),
            "best": df.loc[best_idx, "value"],
            "best_step": int(df.loc[best_idx, "step"]),
        }
    )

by_seed = pd.DataFrame(summary_rows)
by_seed.to_csv(OUT_DIR / "iql_unc_alpha_ablation_summary_by_seed.csv", index=False)

score_seed = by_seed[by_seed["metric"] == "test/target_normalized_score"]
score_summary = (
    score_seed.groupby(["label", "policy"])
    .agg(
        seeds=("seed", "nunique"),
        final_mean=("final", "mean"),
        final_std=("final", "std"),
        best_mean=("best", "mean"),
        best_std=("best", "std"),
    )
    .reset_index()
)

weight_seed = by_seed[by_seed["metric"].str.startswith("train/source_")]
weight_summary = (
    weight_seed.groupby(["label", "metric"])
    .agg(seeds=("seed", "nunique"), mean=("mean", "mean"), std=("mean", "std"))
    .reset_index()
)

score_summary.to_csv(OUT_DIR / "iql_unc_alpha_ablation_score_summary.csv", index=False)
weight_summary.to_csv(OUT_DIR / "iql_unc_alpha_ablation_weight_summary.csv", index=False)

order = [run["label"] for run in RUNS]
score_summary["label"] = pd.Categorical(score_summary["label"], categories=order, ordered=True)
score_summary = score_summary.sort_values("label")

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
axes[0].bar(
    score_summary["label"].astype(str),
    score_summary["final_mean"],
    yerr=score_summary["final_std"],
    capsize=4,
)
axes[0].set_title("Final Target Score")
axes[0].set_ylabel("Normalized score")

axes[1].bar(
    score_summary["label"].astype(str),
    score_summary["best_mean"],
    yerr=score_summary["best_std"],
    capsize=4,
)
axes[1].set_title("Best Target Score")
axes[1].set_ylabel("Normalized score")

for ax in axes:
    ax.tick_params(axis="x", rotation=25)

fig.suptitle("IQL_UNC unc_alpha Ablation | Walker2d Paper-Like | 5 Seeds")
plt.tight_layout()
plt.savefig(OUT_DIR / "iql_unc_alpha_ablation_scores.png", bbox_inches="tight")
plt.close()

weight_mean = weight_summary[weight_summary["metric"] == "train/source_weight_mean"].copy()
weight_mean["label"] = pd.Categorical(weight_mean["label"], categories=order[1:], ordered=True)
weight_mean = weight_mean.sort_values("label")
if not weight_mean.empty:
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=180)
    ax.bar(weight_mean["label"].astype(str), weight_mean["mean"], yerr=weight_mean["std"], capsize=4)
    ax.axhspan(0.3, 0.8, color="green", alpha=0.08)
    ax.set_ylim(0, 1)
    ax.set_title("Mean Source Weight by unc_alpha")
    ax.set_ylabel("Source weight")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "iql_unc_alpha_ablation_weights.png", bbox_inches="tight")
    plt.close()

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(score_summary.to_string(index=False))
print(weight_summary.to_string(index=False))
