import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUT_DIR = Path("analysis_results/cross_env_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TB_CACHE_DIR = OUT_DIR / ".tb_event_cache"
TB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENVS = [
    "hopper-kinematic-legjnt",
    "halfcheetah-kinematic-footjnt",
]

RUNS = [
    {"label": "IQL-target", "policy": "IQL", "root": Path("runs_cross_env_iql_target")},
    {"label": "IQL-mix", "policy": "IQL", "root": Path("runs_cross_env_iql_mix")},
    {"label": "IQL_CONST-w033", "policy": "IQL_UNC", "root": Path("runs_cross_env_iql_const_w033")},
    {"label": "IQL_UNC-a2", "policy": "IQL_UNC", "root": Path("runs_cross_env_iql_unc_a2")},
    {"label": "IQL_BELLMAN_FB", "policy": "IQL_BELLMAN_FB", "root": Path("runs_cross_env_iql_bellman_fb")},
    {"label": "IGDF", "policy": "IGDF", "root": Path("runs_cross_env_igdf")},
]

TAGS = [
    "test/target_normalized_score",
    "train/source_weight_mean",
    "train/source_uncertainty_mean",
    "train/bellman_mismatch_mean",
    "train/source_weight_ess_ratio",
    "train/corr_bellman_mismatch_source_weight",
]


def read_tb(tb_dir, label, policy, env, seed):
    files = sorted(tb_dir.glob("events.out.tfevents*"))
    if not files:
        return []

    event_file = files[-1]
    event_path = str(event_file.resolve())
    if os.name == "nt" and len(event_path) >= 240:
        cached_file = TB_CACHE_DIR / f"{env}_{label}_seed{seed}.tfevents".replace("/", "_")
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
                    "env": env,
                    "label": label,
                    "policy": policy,
                    "seed": seed,
                    "tag": tag,
                    "step": event.step,
                    "value": event.value,
                }
            )
    return rows


rows = []
missing = []
for env in ENVS:
    descriptor = f"{env}-srcdatatype-medium-tardatatype-medium-medium"
    for run in RUNS:
        for seed in range(5):
            tb_dir = run["root"] / run["policy"] / descriptor / f"r{seed}" / "tb"
            run_rows = read_tb(tb_dir, run["label"], run["policy"], env, seed)
            if not run_rows:
                missing.append({"env": env, "label": run["label"], "seed": seed, "tb_dir": str(tb_dir)})
            rows.extend(run_rows)

pd.DataFrame(missing).to_csv(OUT_DIR / "missing_runs.csv", index=False, encoding="utf-8-sig")
if not rows:
    raise SystemExit("No TensorBoard events found.")

scalars = pd.DataFrame(rows)
scalars.to_csv(OUT_DIR / "cross_env_scalars.csv", index=False)

summary_rows = []
for (env, label, policy, seed, tag), df in scalars.groupby(["env", "label", "policy", "seed", "tag"]):
    df = df.sort_values("step")
    values = df["value"]
    best_idx = values.idxmax()
    summary_rows.append(
        {
            "env": env,
            "label": label,
            "policy": policy,
            "seed": seed,
            "metric": tag,
            "final": values.iloc[-1],
            "best": df.loc[best_idx, "value"],
            "best_step": int(df.loc[best_idx, "step"]),
            "last5_avg": values.tail(5).mean(),
            "mean": values.mean(),
            "min": values.min(),
            "max": values.max(),
        }
    )

by_seed = pd.DataFrame(summary_rows)
by_seed.to_csv(OUT_DIR / "cross_env_summary_by_seed.csv", index=False, encoding="utf-8-sig")

score_seed = by_seed[by_seed["metric"] == "test/target_normalized_score"].copy()
score_summary = (
    score_seed.groupby(["env", "label", "policy"])
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

env_order = {env: index for index, env in enumerate(ENVS)}
label_order = {run["label"]: index for index, run in enumerate(RUNS)}
score_summary["env_order"] = score_summary["env"].map(env_order)
score_summary["label_order"] = score_summary["label"].map(label_order)
score_summary = score_summary.sort_values(["env_order", "label_order"]).drop(columns=["env_order", "label_order"])
score_summary.to_csv(OUT_DIR / "cross_env_score_summary.csv", index=False, encoding="utf-8-sig")

paper_table = score_summary.copy()
paper_table["Final"] = paper_table.apply(lambda r: f"{r.final_mean:.2f} +/- {r.final_std:.2f}", axis=1)
paper_table["Best"] = paper_table.apply(lambda r: f"{r.best_mean:.2f} +/- {r.best_std:.2f}", axis=1)
paper_table["Last-5 Avg"] = paper_table.apply(lambda r: f"{r.last5_mean:.2f} +/- {r.last5_std:.2f}", axis=1)
paper_table = paper_table.rename(columns={"env": "Env", "label": "Method", "seeds": "Seeds"})[
    ["Env", "Method", "Seeds", "Final", "Best", "Last-5 Avg"]
]
paper_table.to_csv(OUT_DIR / "cross_env_paper_table.csv", index=False, encoding="utf-8-sig")

paired_frames = []
for env, env_df in score_seed.groupby("env"):
    paired = env_df.pivot_table(index="seed", columns="label", values="last5_avg", aggfunc="first")
    paired = paired.reset_index()
    paired.insert(0, "env", env)
    paired_frames.append(paired)
paired_last5 = pd.concat(paired_frames, ignore_index=True)
paired_last5.to_csv(OUT_DIR / "cross_env_paired_last5.csv", index=False, encoding="utf-8-sig")

win_rows = []
for env, paired in paired_last5.groupby("env"):
    for method in ["IQL-target", "IQL_CONST-w033", "IQL_UNC-a2", "IQL_BELLMAN_FB", "IGDF"]:
        if method not in paired.columns or "IQL-mix" not in paired.columns:
            continue
        valid = paired[[method, "IQL-mix"]].dropna()
        delta = valid[method] - valid["IQL-mix"]
        win_rows.append(
            {
                "env": env,
                "method": method,
                "baseline": "IQL-mix",
                "metric": "last5_avg",
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
                "mean_delta": delta.mean(),
            }
        )
pd.DataFrame(win_rows).to_csv(OUT_DIR / "cross_env_paired_wins_vs_iql_mix.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
for ax, env in zip(axes, ENVS):
    env_summary = score_summary[score_summary["env"] == env].copy()
    x = range(len(env_summary))
    ax.bar(x, env_summary["last5_mean"], yerr=env_summary["last5_std"], capsize=4)
    ax.set_title(env)
    ax.set_xticks(list(x))
    ax.set_xticklabels(env_summary["label"], rotation=25, ha="right")
    ax.set_ylabel("Target normalized score (Last-5 Avg)")
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Cross-Environment Paper-Like 10% Target Comparison")
fig.tight_layout()
fig.savefig(OUT_DIR / "cross_env_last5_comparison.png", dpi=200)
plt.close(fig)

bellman_diag = by_seed[by_seed["label"] == "IQL_BELLMAN_FB"].copy()
if not bellman_diag.empty:
    bellman_diag.to_csv(OUT_DIR / "cross_env_bellman_mechanism_by_seed.csv", index=False, encoding="utf-8-sig")
    mechanism = bellman_diag.pivot_table(index=["env", "seed"], columns="metric", values="mean", aggfunc="first").reset_index()
    mechanism.to_csv(OUT_DIR / "cross_env_bellman_mechanism_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("train/bellman_mismatch_mean", "Bellman Mismatch"),
        ("train/source_weight_ess_ratio", "ESS Ratio"),
        ("train/corr_bellman_mismatch_source_weight", "Corr(Mismatch, Weight)"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        if metric not in mechanism.columns:
            ax.set_visible(False)
            continue
        data = mechanism.groupby("env")[metric].agg(["mean", "std"]).reindex(ENVS)
        x = range(len(data))
        ax.bar(x, data["mean"], yerr=data["std"], capsize=4)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels([env.split("-")[0] for env in data.index], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("IQL_BELLMAN_FB Mechanism Diagnostics")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_env_bellman_mechanism.png", dpi=200)
    plt.close(fig)

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(paper_table.to_string(index=False))
print("\nPaired wins vs IQL-mix, Last-5 Avg:")
print(pd.DataFrame(win_rows).to_string(index=False))
