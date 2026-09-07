"""Aggregate Timer RoPE-rebase and adaptive-refresh experiment JSON files."""

import argparse
import glob
import json
import os
import statistics


def median(values):
    return float(statistics.median(values))


def mean(values):
    return float(statistics.fmean(values))


def load_by_dataset(pattern):
    return {
        os.path.basename(path).split("_L")[0]: json.load(open(path))
        for path in glob.glob(pattern)
    }


def summarize_methods(datasets, method):
    rows = [document["methods"][method] for document in datasets.values()]
    baselines = [document["methods"]["1"] for document in datasets.values()]
    deltas = [row["forecast_mae_delta_vs_full_k1"] for row in rows]
    relative_deltas = [delta / base["mae"] for delta, base in zip(deltas, baselines)]
    relative_gaps = [
        row["prediction_gap_mae_vs_full_k1"] / base["mae"]
        for row, base in zip(rows, baselines)
    ]
    summary = {
        "datasets": len(rows),
        "median_latency_ms": median([row["mean_latency_ms"] for row in rows]),
        "median_speedup_vs_k1": median([
            base["mean_latency_ms"] / row["mean_latency_ms"]
            for row, base in zip(rows, baselines)
        ]),
        "mean_mae_delta": mean(deltas),
        "median_mae_delta": median(deltas),
        "median_relative_mae_delta": median(relative_deltas),
        "median_relative_prediction_gap": median(relative_gaps),
        "mae_wins_vs_k1": sum(delta < 0 for delta in deltas),
    }
    if "full_graph_replays" in rows[0]:
        summary["total_full_graph_replays"] = sum(
            row["full_graph_replays"] for row in rows
        )
    return summary


def main(args):
    fixed = {}
    for mode in ("baseline", "rope_rebase"):
        datasets = load_by_dataset(
            os.path.join(args.fixed_root, mode, "*_L2880.json")
        )
        fixed[mode] = {
            method: summarize_methods(datasets, method)
            for method in ("1", "2", "3", "4", "16", "0")
        }

    rebase_minus_baseline = {}
    baseline = load_by_dataset(
        os.path.join(args.fixed_root, "baseline", "*_L2880.json")
    )
    rebased = load_by_dataset(
        os.path.join(args.fixed_root, "rope_rebase", "*_L2880.json")
    )
    for method in ("2", "3", "4", "16", "0"):
        deltas = [
            rebased[name]["methods"][method]["mae"]
            - baseline[name]["methods"][method]["mae"]
            for name in sorted(baseline)
        ]
        rebase_minus_baseline[method] = {
            "mean_mae_change": mean(deltas),
            "median_mae_change": median(deltas),
            "datasets_improved": sum(delta < 0 for delta in deltas),
        }

    adaptive_by_context = {}
    for context_length in (480, 960, 1920, 2880):
        datasets = load_by_dataset(
            os.path.join(
                args.adaptive_root, "rope_rebase", f"*_L{context_length}.json"
            )
        )
        adaptive_by_context[str(context_length)] = summarize_methods(
            datasets, "adaptive_0.2"
        )

    datasets_2880 = load_by_dataset(
        os.path.join(args.adaptive_root, "rope_rebase", "*_L2880.json")
    )
    threshold_sweep = {
        threshold: summarize_methods(datasets_2880, threshold)
        for threshold in (
            "adaptive_0.1",
            "adaptive_0.2",
            "adaptive_0.3",
            "adaptive_0.5",
        )
    }
    output = {
        "fixed_refresh_l2880": fixed,
        "rope_rebase_minus_baseline_l2880": rebase_minus_baseline,
        "adaptive_threshold_sweep_l2880": threshold_sweep,
        "adaptive_0.2_by_context": adaptive_by_context,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--adaptive-root", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
