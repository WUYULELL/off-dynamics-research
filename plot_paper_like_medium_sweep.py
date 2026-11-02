import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUT_DIR = Path("analysis_results/paper_like_medium_sweep")
RUN_ROOT = Path("runs_paper_like_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAGS = [
    "test/target_return",
    "test/target_normalized_score",
    "test/source_return",
]


def read_scalar(event_file: Path, tag: str, meta: dict) -> pd.DataFrame:
    acc = EventAccumulator(str(event_file))
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                **meta,
                "step": event.step,
                "value": event.value,
                "tag": tag,
            }
            for event in acc.Scalars(tag)
        ]
    )


def parse_run_dir(path: Path) -> dict:
    # Example:
    # runs_paper_like_sweep/IGDF/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb
    policy = path.parts[-4]
    descriptor = path.parts[-3]
    seed = int(path.parts[-2].lstrip("r"))
    env_name, rest = descriptor.split("-srcdatatype-", 1)
    src_type, rest = rest.split("-tardatatype-", 1)
    tar_type, shift_level = rest.rsplit("-", 1)
    return {
        "policy": policy,
        "env": env_name,
        "srctype": src_type,
        "tartype": tar_type,
        "shift_level": shift_level,
        "seed": seed,
    }


frames = []
for tb_dir in RUN_ROOT.glob("*/*/r*/tb"):
    event_files = sorted(tb_dir.glob("events.out.tfevents*"))
    if not event_files:
        continue
    meta = parse_run_dir(tb_dir)
    event_file = event_files[-1]
    for tag in TAGS:
        df = read_scalar(event_file, tag, meta)
        if not df.empty:
            frames.append(df)

if not frames:
    raise SystemExit(f"No TensorBoard event files found under {RUN_ROOT.resolve()}")

scalars = pd.concat(frames, ignore_index=True)
scalars.to_csv(OUT_DIR / "paper_like_medium_sweep_scalars.csv", index=False)

summary_rows = []
for keys, df in scalars.groupby(["policy", "env", "shift_level", "srctype", "tartype", "seed", "tag"]):
    policy, env, shift_level, src_type, tar_type, seed, tag = keys
    df = df.sort_values("step")
    best = df.loc[df["value"].idxmax()]
    summary_rows.append(
        {
            "policy": policy,
            "env": env,
            "shift_level": shift_level,
            "srctype": src_type,
            "tartype": tar_type,
            "seed": seed,
            "metric": tag,
            "initial": df.iloc[0]["value"],
            "final": df.iloc[-1]["value"],
            "best": best["value"],
            "best_step": int(best["step"]),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "paper_like_medium_sweep_summary_by_seed.csv", index=False)

target_summary = summary[
    summary["metric"].isin(["test/target_return", "test/target_normalized_score"])
].copy()

agg = (
    target_summary.groupby(["policy", "env", "metric"])
    .agg(
        final_mean=("final", "mean"),
        final_std=("final", "std"),
        best_mean=("best", "mean"),
        best_std=("best", "std"),
        seeds=("seed", "nunique"),
    )
    .reset_index()
)
agg["final_std"] = agg["final_std"].fillna(0.0)
agg["best_std"] = agg["best_std"].fillna(0.0)
agg.to_csv(OUT_DIR / "paper_like_medium_sweep_summary.csv", index=False)

score = agg[agg["metric"] == "test/target_normalized_score"].copy()
if not score.empty:
    plt.style.use("seaborn-v0_8-whitegrid")
    pivot = score.pivot(index="env", columns="policy", values="best_mean")
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=180)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Best Target Normalized Score | Paper-Like 10% Target")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Best normalized score")
    ax.legend(frameon=False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "best_target_normalized_score_bar.png", bbox_inches="tight")
    plt.close()

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(agg.sort_values(["metric", "env", "policy"]).to_string(index=False))
