import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_DIR = Path("runs/IQL/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb")
OUT_DIR = Path("analysis_results/iql_naive_mix_walker2d")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_scalar(event_file: Path, tag: str) -> pd.DataFrame:
    acc = EventAccumulator(str(event_file))
    acc.Reload()
    rows = [
        {"step": event.step, "value": event.value, "wall_time": event.wall_time}
        for event in acc.Scalars(tag)
    ]
    df = pd.DataFrame(rows)
    df["tag"] = tag
    return df


def smooth(values: pd.Series, window: int = 5) -> pd.Series:
    return values.rolling(window=window, min_periods=1, center=True).mean()


event_files = sorted(RUN_DIR.glob("events.out.tfevents*"))
if not event_files:
    raise SystemExit(f"No TensorBoard event files found under {RUN_DIR}")

event_file = event_files[-1]
tags = [
    "test/source_return",
    "test/target_return",
    "test/target_normalized_score",
]

frames = [read_scalar(event_file, tag) for tag in tags]
all_scalars = pd.concat(frames, ignore_index=True)
all_scalars.to_csv(OUT_DIR / "iql_naive_mix_scalars.csv", index=False)

wide = all_scalars.pivot_table(index="step", columns="tag", values="value").reset_index()
wide.to_csv(OUT_DIR / "iql_naive_mix_eval_curve.csv", index=False)

summary_rows = []
for tag in tags:
    df = all_scalars[all_scalars["tag"] == tag].sort_values("step")
    best_idx = df["value"].idxmax()
    final = df.iloc[-1]
    best = df.loc[best_idx]
    summary_rows.append(
        {
            "metric": tag,
            "initial": df.iloc[0]["value"],
            "final": final["value"],
            "best": best["value"],
            "best_step": int(best["step"]),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "iql_naive_mix_summary.csv", index=False)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=180)

ax = axes[0]
for tag, label, color in [
    ("test/source_return", "Source return", "#4C78A8"),
    ("test/target_return", "Target return", "#F58518"),
]:
    df = all_scalars[all_scalars["tag"] == tag].sort_values("step")
    ax.plot(df["step"], df["value"], color=color, alpha=0.28, linewidth=1)
    ax.plot(df["step"], smooth(df["value"]), color=color, linewidth=2.2, label=label)
ax.set_title("IQL Naive-Mix Evaluation Return")
ax.set_xlabel("Gradient steps")
ax.set_ylabel("Average return over 10 episodes")
ax.legend(frameon=False)

ax = axes[1]
df = all_scalars[all_scalars["tag"] == "test/target_normalized_score"].sort_values("step")
ax.plot(df["step"], df["value"], color="#54A24B", alpha=0.30, linewidth=1)
ax.plot(df["step"], smooth(df["value"]), color="#54A24B", linewidth=2.2)
ax.set_title("Target Normalized Score")
ax.set_xlabel("Gradient steps")
ax.set_ylabel("Normalized score")

fig.suptitle(
    "Walker2d Kinematic-FootJnt Medium | Source Medium + Target Medium | Seed 0",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "iql_naive_mix_curves.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 2.4), dpi=180)
ax.axis("off")
table_df = summary.copy()
for col in ["initial", "final", "best"]:
    table_df[col] = table_df[col].map(lambda x: f"{x:.2f}")
table_df["best_step"] = table_df["best_step"].map(lambda x: f"{x:,}")
table_df["metric"] = table_df["metric"].map(
    {
        "test/source_return": "Source return",
        "test/target_return": "Target return",
        "test/target_normalized_score": "Target normalized score",
    }
)
table = ax.table(
    cellText=table_df[["metric", "initial", "final", "best", "best_step"]].values,
    colLabels=["Metric", "Initial", "Final", "Best", "Best step"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.35)
ax.set_title("IQL Naive-Mix Summary", fontsize=12, pad=10)
fig.savefig(OUT_DIR / "iql_naive_mix_summary_table.png", bbox_inches="tight")
plt.close(fig)

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(summary.to_string(index=False))
