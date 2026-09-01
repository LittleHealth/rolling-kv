"""Aggregate the final Sundial rolling-cache experiment matrix."""

import argparse
import glob
import json
import os
import statistics


def median(values):
    return float(statistics.median(values))


def mean(values):
    return float(statistics.fmean(values))


def load(pattern):
    return [json.load(open(path)) for path in glob.glob(pattern)]


def aggregate(documents, method):
    rows = [document["methods"][method] for document in documents]
    baselines = [document["methods"]["1"] for document in documents]
    relative_deltas = [
        row["forecast_mae_delta_vs_full_k1"] / baseline["mae"]
        for row, baseline in zip(rows, baselines)
    ]
    return {
        "datasets": len(rows),
        "median_full_latency_ms": median(
            baseline["mean_latency_ms"] for baseline in baselines
        ),
        "median_method_latency_ms": median(row["mean_latency_ms"] for row in rows),
        "median_speedup": median(
            baseline["mean_latency_ms"] / row["mean_latency_ms"]
            for row, baseline in zip(rows, baselines)
        ),
        "median_relative_prediction_gap": median(
            row["prediction_gap_mae_vs_full_k1"] / baseline["mae"]
            for row, baseline in zip(rows, baselines)
        ),
        "median_relative_mae_delta": median(relative_deltas),
        "mean_relative_mae_delta": mean(relative_deltas),
        "mae_wins_vs_full": sum(value < 0 for value in relative_deltas),
    }


def main(args):
    recommended = {}
    for context_length in (480, 960, 1920, 2880):
        documents = load(
            os.path.join(
                args.results_root,
                "sundial_antithetic_s5_t10_0812/rebase",
                f"*_L{context_length}.json",
            )
        )
        recommended[str(context_length)] = {
            method: aggregate(documents, method) for method in ("4", "16")
        }

    adaptive_documents = load(
        os.path.join(
            args.results_root,
            "sundial_adaptive_s5_t10_0812/rebase/*_L2880.json",
        )
    )
    adaptive = {
        method: aggregate(adaptive_documents, method)
        for method in ("adaptive_0.05", "adaptive_0.1", "adaptive_0.2")
    }
    output = {
        "model": "Sundial-base-128m",
        "checkpoint_revision": "3212e42564493f520593e5414af4367fc4b49226",
        "recommended_protocol": {
            "num_samples": 5,
            "sampling_steps": 10,
            "noise_mode": "antithetic",
            "rope_rebase": True,
            "default_refresh_length": 4,
            "performance_refresh_length": 16,
        },
        "recommended_by_context": recommended,
        "adaptive_l2880": adaptive,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())

