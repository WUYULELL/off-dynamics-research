import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUT_DIR = Path("analysis_results/baseline_comparison_walker2d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "IQL naive-mix": Path(
        "runs/IQL/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb"
    ),
    "TD3_BC naive-mix": Path(
        "runs/TD3_BC/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb"
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
                "wall_time": event.wall_time,
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
        raise SystemExit(f"No TensorBoard event files found for {method}: {run_dir}")
    event_file = event_files[-1]
    for tag in TAGS:
        frames.append(read_scalar(event_file, tag, method))

all_scalars = pd.concat(frames, ignore_index=True)
all_scalars.to_csv(OUT_DIR / "baseline_comparison_scalars.csv", index=False)

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
summary.to_csv(OUT_DIR / "baseline_comparison_summary.csv", index=False)

plt.style.use("seaborn-v0_8-whitegrid")
colors = {
    "IQL naive-mix": "#4C78A8",
    "TD3_BC naive-mix": "#F58518",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=180)

ax = axes[0]
for method in RUNS:
    df = all_scalars[
        (all_scalars["method"] == method) & (all_scalars["tag"] == "test/target_return")
    ].sort_values("step")
    ax.plot(df["step"], df["value"], color=colors[method], alpha=0.25, linewidth=1)
    ax.plot(df["step"], smooth(df["value"]), color=colors[method], linewidth=2.2, label=method)
ax.set_title("Target Return")
ax.set_xlabel("Gradient steps")
ax.set_ylabel("Average return over 10 episodes")
ax.legend(frameon=False)

ax = axes[1]
for method in RUNS:
    df = all_scalars[
        (all_scalars["method"] == method)
        & (all_scalars["tag"] == "test/target_normalized_score")
    ].sort_values("step")
    ax.plot(df["step"], df["value"], color=colors[method], alpha=0.25, linewidth=1)
    ax.plot(df["step"], smooth(df["value"]), color=colors[method], linewidth=2.2, label=method)
ax.set_title("Target Normalized Score")
ax.set_xlabel("Gradient steps")
ax.set_ylabel("Normalized score")
ax.legend(frameon=False)

fig.suptitle(
    "Walker2d Kinematic-FootJnt Medium | Source Medium + Target Medium | Seed 0",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "baseline_comparison_curves.png", bbox_inches="tight")
plt.close(fig)

target_summary = summary[summary["metric"].isin(["test/target_return", "test/target_normalized_score"])].copy()
target_summary["metric"] = target_summary["metric"].map(
    {
        "test/target_return": "Target return",
        "test/target_normalized_score": "Target normalized score",
    }
)

fig, ax = plt.subplots(figsize=(10, 2.8), dpi=180)
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
table.set_fontsize(8.3)
table.scale(1, 1.35)
ax.set_title("Naive-Mix Baseline Comparison", fontsize=12, pad=10)
fig.savefig(OUT_DIR / "baseline_comparison_summary_table.png", bbox_inches="tight")
plt.close(fig)

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(target_summary.sort_values(["metric", "method"]).to_string(index=False))
