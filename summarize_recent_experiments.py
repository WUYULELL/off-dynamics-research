from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ENVS = ["hopper-kinematic-legjnt", "halfcheetah-kinematic-footjnt"]


def read_tb(tb_dir: Path):
    files = sorted(tb_dir.glob("events.out.tfevents*"))
    if not files:
        return None

    acc = EventAccumulator(str(files[-1]))
    acc.Reload()
    tags = acc.Tags().get("scalars", [])

    score_tag = None
    candidates = [
        "test/target_normalized_score",
        "eval/target_normalized_score",
        "target_normalized_score",
    ]
    for tag in candidates:
        if tag in tags:
            score_tag = tag
            break
    if score_tag is None:
        matches = [
            tag
            for tag in tags
            if "target" in tag.lower()
            and "normalized" in tag.lower()
            and "score" in tag.lower()
        ]
        score_tag = matches[0] if matches else None
    if score_tag is None:
        return {"error": "no target normalized score tag", "tags": tags}

    vals = [x.value for x in acc.Scalars(score_tag)]
    out = {
        "n_eval": len(vals),
        "final": vals[-1],
        "best": max(vals),
        "last5": sum(vals[-5:]) / min(5, len(vals)),
    }

    for tag in [
        "train/source_weight_mean",
        "train/source_weight_ess_ratio",
        "train/source_weight_at_min_ratio",
        "train/bellman_mismatch_mean",
        "train/corr_bellman_mismatch_source_weight",
    ]:
        if tag in tags:
            out[tag.split("/")[-1]] = acc.Scalars(tag)[-1].value
    return out


def print_rows(title, rows):
    print("\n" + title)
    print(
        "env,label,n_eval,final,best,last5,source_weight_mean,"
        "source_weight_ess_ratio,source_weight_at_min_ratio,"
        "bellman_mismatch_mean,corr_bellman_mismatch_source_weight"
    )
    for env, label, tb_dir in rows:
        result = read_tb(tb_dir)
        if result is None:
            print(f"{env},{label},MISSING")
            continue
        if "error" in result:
            print(f"{env},{label},ERROR,{result['error']}")
            continue
        print(
            ",".join(
                [
                    env,
                    label,
                    str(result.get("n_eval", "")),
                    f"{result.get('final', float('nan')):.4f}",
                    f"{result.get('best', float('nan')):.4f}",
                    f"{result.get('last5', float('nan')):.4f}",
                    _fmt(result.get("source_weight_mean")),
                    _fmt(result.get("source_weight_ess_ratio")),
                    _fmt(result.get("source_weight_at_min_ratio")),
                    _fmt(result.get("bellman_mismatch_mean")),
                    _fmt(result.get("corr_bellman_mismatch_source_weight")),
                ]
            )
        )


def _fmt(value):
    return "" if value is None else f"{value:.4f}"


def main():
    budget_rows = []
    for env in ENVS:
        for budget in ["b0p10", "b0p20", "b0p33", "b0p50"]:
            tb_dir = (
                Path("runs_budget_ablation_bellman_fb")
                / budget
                / "IQL_BELLMAN_FB"
                / f"{env}-srcdatatype-medium-tardatatype-medium-medium"
                / "r0"
                / "tb"
            )
            budget_rows.append((env, f"IQL_BELLMAN_FB-{budget}", tb_dir))

    target5_rows = []
    methods = [
        ("IQL-target-5pct", "runs_5pct_iql_target", "IQL"),
        ("IQL-mix-5pct", "runs_5pct_iql_mix", "IQL"),
        ("IQL_CONST-w033-5pct", "runs_5pct_iql_const_w033", "IQL_UNC"),
        ("IQL_BELLMAN_FB-5pct", "runs_5pct_iql_bellman_fb", "IQL_BELLMAN_FB"),
    ]
    for env in ENVS:
        for label, root, policy in methods:
            tb_dir = (
                Path(root)
                / policy
                / f"{env}-srcdatatype-medium-tardatatype-medium-medium"
                / "r0"
                / "tb"
            )
            target5_rows.append((env, label, tb_dir))

    print_rows("Budget ablation, 10pct target, seed 0", budget_rows)
    print_rows("5pct target core comparison, seed 0", target5_rows)


if __name__ == "__main__":
    main()
