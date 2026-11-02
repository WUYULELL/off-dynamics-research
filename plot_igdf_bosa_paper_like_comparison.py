import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUT_DIR = Path("analysis_results/igdf_bosa_paper_like_comparison_walker2d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "BOSA paper-like 10% target": Path(
        "runs_paper_like/BOSA/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb"
    ),
    "IGDF paper-like 10% target": Path(
        "runs_paper_like/IGDF/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb"
    ),
}

TAGS = [
    "test/source_return",
    "test/target_return",
    "test/target_normalized_score",
]


def latest_event_file(run_dir: Path, method: str) -> Path:
    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    if not event_files:
        raise SystemExit(
            f"No TensorBoard event file found for {method}: {run_dir}\n"
            "Run the corresponding experiment script first."
        )
    return event_files[-1]


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
    event_file = latest_event_file(run_dir, method)
    for tag in TAGS:
        frames.append(read_scalar(event_file, tag, method))

all_scalars = pd.concat(frames, ignore_index=True)
all_scalars.to_csv(OUT_DIR / "igdf_bosa_paper_like_scalars.csv", index=False)

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
summary.to_csv(OUT_DIR / "igdf_bosa_paper_like_summary.csv", index=False)

plt.style.use("seaborn-v0_8-whitegrid")
colors = {
    "BOSA paper-like 10% target": "#54A24B",
    "IGDF paper-like 10% target": "#B279A2",
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
        ax.plot(df["step"], smooth(df["value"]), color=colors[method], linewidth=2.2, label=method)
    ax.set_title(title)
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)

fig.suptitle(
    "Paper-Like 10% Target Comparison | Walker2d Kinematic-FootJnt Medium | Seed 0",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "igdf_bosa_paper_like_curves.png", bbox_inches="tight")
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

fig, ax = plt.subplots(figsize=(10.5, 3.0), dpi=180)
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
table.set_fontsize(8.2)
table.scale(1, 1.35)
ax.set_title("Paper-Like 10% Target Summary", fontsize=12, pad=10)
fig.savefig(OUT_DIR / "igdf_bosa_paper_like_summary_table.png", bbox_inches="tight")
plt.close(fig)

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(target_summary.sort_values(["metric", "method"]).to_string(index=False))
