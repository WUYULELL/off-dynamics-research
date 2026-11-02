import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_results/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("analysis_results/iql_unc_vs_bosa_igdf_existing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

iql_summary = pd.read_csv("analysis_results/iql_unc_10pct_comparison/iql_unc_10pct_summary.csv")
paper_summary = pd.read_csv("analysis_results/paper_like_medium_sweep/paper_like_medium_sweep_summary.csv")

rows = []

for method in ["IQL mix 10% target", "IQL_UNC 10% target"]:
    score = iql_summary[
        (iql_summary["method"] == method)
        & (iql_summary["metric"] == "test/target_normalized_score")
    ].iloc[0]
    ret = iql_summary[
        (iql_summary["method"] == method)
        & (iql_summary["metric"] == "test/target_return")
    ].iloc[0]
    rows.append(
        {
            "method": method,
            "protocol": "ODRL default direction, seed 0",
            "seeds": 1,
            "final_score_mean": score["final"],
            "final_score_std": 0.0,
            "best_score_mean": score["best"],
            "best_score_std": 0.0,
            "final_return_mean": ret["final"],
            "final_return_std": 0.0,
            "best_return_mean": ret["best"],
            "best_return_std": 0.0,
        }
    )

for method in ["BOSA", "IGDF"]:
    score = paper_summary[
        (paper_summary["policy"] == method)
        & (paper_summary["env"] == "walker2d-kinematic-footjnt")
        & (paper_summary["metric"] == "test/target_normalized_score")
    ].iloc[0]
    ret = paper_summary[
        (paper_summary["policy"] == method)
        & (paper_summary["env"] == "walker2d-kinematic-footjnt")
        & (paper_summary["metric"] == "test/target_return")
    ].iloc[0]
    rows.append(
        {
            "method": method,
            "protocol": "paper-like direction, 5 seeds",
            "seeds": int(score["seeds"]),
            "final_score_mean": score["final_mean"],
            "final_score_std": score["final_std"],
            "best_score_mean": score["best_mean"],
            "best_score_std": score["best_std"],
            "final_return_mean": ret["final_mean"],
            "final_return_std": ret["final_std"],
            "best_return_mean": ret["best_mean"],
            "best_return_std": ret["best_std"],
        }
    )

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "iql_unc_vs_bosa_igdf_existing_summary.csv", index=False)

plt.style.use("seaborn-v0_8-whitegrid")
colors = {
    "IQL mix 10% target": "#4C78A8",
    "IQL_UNC 10% target": "#B279A2",
    "BOSA": "#54A24B",
    "IGDF": "#E45756",
}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=180)
x = range(len(summary))
labels = summary["method"].tolist()

axes[0].bar(
    x,
    summary["final_score_mean"],
    yerr=summary["final_score_std"],
    capsize=4,
    color=[colors[m] for m in labels],
)
axes[0].set_title("Final Target Normalized Score")
axes[0].set_ylabel("Normalized score")
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(labels, rotation=20, ha="right")

axes[1].bar(
    x,
    summary["best_score_mean"],
    yerr=summary["best_score_std"],
    capsize=4,
    color=[colors[m] for m in labels],
)
axes[1].set_title("Best Target Normalized Score")
axes[1].set_ylabel("Normalized score")
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(labels, rotation=20, ha="right")

fig.suptitle(
    "Walker2d Kinematic-FootJnt Medium | Existing Runs\n"
    "Note: IQL/IQL_UNC use ODRL default direction seed 0; BOSA/IGDF use paper-like direction 5 seeds.",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "iql_unc_vs_bosa_igdf_existing_bar.png", bbox_inches="tight")
plt.close(fig)

print(f"Wrote results to: {OUT_DIR.resolve()}")
print(summary.to_string(index=False))
