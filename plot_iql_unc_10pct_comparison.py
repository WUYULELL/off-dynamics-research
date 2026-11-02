import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUT_DIR = Path("analysis_results/iql_unc_10pct_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "IQL mix 10% target": Path(
        "runs_10pct/IQL/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb"
    ),
    "IQL_UNC 10% target": Path(
        "runs_10pct/IQL_UNC/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb"
    ),
}

TAGS = [
    "test/target_return",
    "test/target_normalized_score",
    "test/source_return",
]


def read_scalar(event_file: Path, tag: str, method: str) -> pd.DataFrame:
    acc = EventAccumulator(str(event_file))
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {"method": method, "step": event.step, "value": event.value, "tag": tag}
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
    for tag in TAGS:
        df = read_scalar(event_files[-1], tag, method)
        if not df.empty:
            frames.append(df)

all_scalars = pd.concat(frames, ignore_index=True)
all_scalars.to_csv(OUT_DIR / "iql_unc_10pct_scalars.csv", index=False)

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
            "last_step": int(df.iloc[-1]["step"]),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "iql_unc_10pct_summary.csv", index=False)

plt.style.use("seaborn-v0_8-whitegrid")
colors = {
    "IQL mix 10% target": "#4C78A8",
    "IQL_UNC 10% target": "#B279A2",
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
    "IQL Mix vs IQL_UNC | 100% Source + 10% Target | Walker2d Kinematic-FootJnt Medium | Seed 0",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "iql_unc_10pct_curves.png", bbox_inches="tight")
plt.close(fig)

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(summary[summary["metric"].isin(["test/target_return", "test/target_normalized_score"])].to_string(index=False))
