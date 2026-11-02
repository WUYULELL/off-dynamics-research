import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUT_DIR = Path("analysis_results/paper_like_iql_unc_vs_bosa_igdf_walker2d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ROOTS = {
    "IQL": Path("runs_paper_like_iql_unc"),
    "IQL_UNC": Path("runs_paper_like_iql_unc"),
    "BOSA": Path("runs_paper_like_sweep"),
    "IGDF": Path("runs_paper_like_sweep"),
}

ENV_NAME = "walker2d-kinematic-footjnt"
METRICS = ["test/target_return", "test/target_normalized_score"]


def parse_run_dir(path: Path) -> dict:
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


def read_scalar(tb_dir: Path, tag: str, meta: dict) -> pd.DataFrame:
    event_files = sorted(tb_dir.glob("events.out.tfevents*"))
    if not event_files:
        return pd.DataFrame()

    acc = EventAccumulator(str(event_files[-1]))
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                **meta,
                "step": event.step,
                "value": event.value,
                "metric": tag,
            }
            for event in acc.Scalars(tag)
        ]
    )


frames = []
for policy, root in RUN_ROOTS.items():
    if not root.exists():
        continue
    for tb_dir in root.glob(f"{policy}/*/r*/tb"):
        meta = parse_run_dir(tb_dir)
        if meta["env"] != ENV_NAME:
            continue
        for metric in METRICS:
            df = read_scalar(tb_dir, metric, meta)
            if not df.empty:
                frames.append(df)

if not frames:
    raise SystemExit("No TensorBoard scalars found. Run run_iql_iql_unc_paper_like_walker2d_5seeds.ps1 first.")

scalars = pd.concat(frames, ignore_index=True)
scalars.to_csv(OUT_DIR / "paper_like_iql_unc_vs_bosa_igdf_walker2d_scalars.csv", index=False)

summary_rows = []
for keys, df in scalars.groupby(["policy", "env", "seed", "metric"]):
    policy, env_name, seed, metric = keys
    df = df.sort_values("step")
    best = df.loc[df["value"].idxmax()]
    summary_rows.append(
        {
            "policy": policy,
            "env": env_name,
            "seed": seed,
            "metric": metric,
            "initial": df.iloc[0]["value"],
            "final": df.iloc[-1]["value"],
            "best": best["value"],
            "best_step": int(best["step"]),
        }
    )

summary_by_seed = pd.DataFrame(summary_rows)
summary_by_seed.to_csv(OUT_DIR / "paper_like_iql_unc_vs_bosa_igdf_walker2d_summary_by_seed.csv", index=False)

summary = (
    summary_by_seed.groupby(["policy", "env", "metric"])
    .agg(
        seeds=("seed", "nunique"),
        final_mean=("final", "mean"),
        final_std=("final", "std"),
        best_mean=("best", "mean"),
        best_std=("best", "std"),
    )
    .reset_index()
)
summary["final_std"] = summary["final_std"].fillna(0.0)
summary["best_std"] = summary["best_std"].fillna(0.0)
summary.to_csv(OUT_DIR / "paper_like_iql_unc_vs_bosa_igdf_walker2d_summary.csv", index=False)

score = summary[summary["metric"] == "test/target_normalized_score"].copy()
policy_order = ["IQL", "IQL_UNC", "BOSA", "IGDF"]
score["policy"] = pd.Categorical(score["policy"], categories=policy_order, ordered=True)
score = score.sort_values("policy")

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=180)

axes[0].bar(score["policy"].astype(str), score["final_mean"], yerr=score["final_std"], capsize=4)
axes[0].set_title("Final Target Score")
axes[0].set_ylabel("Normalized score")
axes[0].set_xlabel("Method")

axes[1].bar(score["policy"].astype(str), score["best_mean"], yerr=score["best_std"], capsize=4)
axes[1].set_title("Best Target Score")
axes[1].set_ylabel("Normalized score")
axes[1].set_xlabel("Method")

for ax in axes:
    ax.tick_params(axis="x", rotation=20)

fig.suptitle("Walker2d Paper-Like 10% Target | 5 Seeds")
plt.tight_layout()
plt.savefig(OUT_DIR / "paper_like_iql_unc_vs_bosa_igdf_walker2d_bar.png", bbox_inches="tight")
plt.close()

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(summary.sort_values(["metric", "policy"]).to_string(index=False))
