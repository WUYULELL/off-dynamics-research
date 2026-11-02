import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUT_DIR = Path("analysis_results/target_only_vs_naive_mix_walker2d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "IQL naive-mix": Path(
        "runs/IQL/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb"
    ),
    "IQL target-only": Path(
        "runs_target_only/IQL/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb"
    ),
    "TD3_BC naive-mix": Path(
        "runs/TD3_BC/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb"
    ),
    "TD3_BC target-only": Path(
        "runs_target_only/TD3_BC/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb"
    ),
}

TAGS = [
    "test/source_return",
    "test/target_return",
    "test/target_normalized_score",
]


def read_scalar(event_file: Path, tag: str, method: str) -> pd.DataFrame:
    acc = EventAccumulator(str(event_file))
    acc.Reload()
    return pd.DataFrame(
        [
            {
                "method": method,
                "step": event.step,
                "value": event.value,
                "tag": tag,
            }
            for event in acc.Scalars(tag)
        ]
    )


def smooth(values: pd.Series, window: int = 5) -> pd.Series:
    return values.rolling(window=window, min_periods=1, center=True).mean()


frames = []
for method, run_dir in RUNS.items():
    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    if not event_files:
        raise SystemExit(f"No TensorBoard event file found for {method}: {run_dir}")
    event_file = event_files[-1]
    for tag in TAGS:
        frames.append(read_scalar(event_file, tag, method))

all_scalars = pd.concat(frames, ignore_index=True)
all_scalars.to_csv(OUT_DIR / "target_only_vs_naive_mix_scalars.csv", index=False)

summary_rows = []
for (method, tag), df in all_scalars.groupby(["method", "tag"]):
    df = df.sort_values("step")
    best = df.loc[df["value"].idxmax()]
    summary_rows.append(
        {
            "method": method,
            "metric": tag,
            "initial": df.iloc[0]["value"],
            "final": df.iloc[-1]["value"],
            "best": best["value"],
            "best_step": int(best["step"]),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "target_only_vs_naive_mix_summary.csv", index=False)

plt.style.use("seaborn-v0_8-whitegrid")
colors = {
    "IQL naive-mix": "#4C78A8",
    "IQL target-only": "#72B7B2",
    "TD3_BC naive-mix": "#F58518",
    "TD3_BC target-only": "#E45756",
}
linestyles = {
    "IQL naive-mix": "-",
    "IQL target-only": "--",
    "TD3_BC naive-mix": "-",
    "TD3_BC target-only": "--",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=180)

for ax, tag, title, ylabel in [
    (axes[0], "test/target_return", "Target Return", "Average return over 10 episodes"),
    (axes[1], "test/target_normalized_score", "Target Normalized Score", "Normalized score"),
]:
    for method in RUNS:
        df = all_scalars[
            (all_scalars["method"] == method) & (all_scalars["tag"] == tag)
        ].sort_values("step")
        ax.plot(df["step"], df["value"], color=colors[method], alpha=0.18, linewidth=1)
        ax.plot(
            df["step"],
            smooth(df["value"]),
            color=colors[method],
            linestyle=linestyles[method],
            linewidth=2.2,
            label=method,
        )
    ax.set_title(title)
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)

fig.suptitle(
    "Target-Only vs Naive-Mix | Walker2d Kinematic-FootJnt Medium | Seed 0",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "target_only_vs_naive_mix_curves.png", bbox_inches="tight")
plt.close(fig)

target_summary = summary[
    summary["metric"].isin(["test/target_return", "test/target_normalized_score"])
].copy()
target_summary["metric"] = target_summary["metric"].map(
    {
        "test/target_return": "Target return",
        "test/target_normalized_score": "Target normalized score",
    }
)

fig, ax = plt.subplots(figsize=(11.5, 3.4), dpi=180)
ax.axis("off")
table_df = target_summary.sort_values(["metric", "method"]).copy()
for col in ["initial", "final", "best"]:
    table_df[col] = table_df[col].map(lambda x: f"{x:.2f}")
table_df["best_step"] = table_df["best_step"].map(lambda x: f"{x:,}")
table = ax.table(
    cellText=table_df[["method", "metric", "initial", "final", "best", "best_step"]].values,
    colLabels=["Method", "Metric", "Initial", "Final", "Best", "Best step"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.35)
ax.set_title("Target-Only vs Naive-Mix Summary", fontsize=12, pad=10)
fig.savefig(OUT_DIR / "target_only_vs_naive_mix_summary_table.png", bbox_inches="tight")
plt.close(fig)

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(target_summary.sort_values(["metric", "method"]).to_string(index=False))
