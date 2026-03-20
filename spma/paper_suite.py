from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig, build_default_config
from .main import run_experiment
from .utils import ensure_dir, markdown_table, save_csv, save_json

DEFAULT_BENCHMARKS = [
    "split_cifar100",
    "split_tiny_imagenet",
    "tiny_imagenet_compatible_shift",
]

DEFAULT_METHODS = [
    "plain_ft",
    "kd_only",
    "er_replay_512",
    "spma_sparse_smooth_boosted_512",
    "spma_sparse_smooth_decoupled_factor_512",
    "spma_manifold_continuation_balanced_512",
]

DEFAULT_SEEDS = [7, 11, 13, 17, 19]

AGGREGATE_METRICS = [
    "old_accuracy_before_ft",
    "old_accuracy_after_ft",
    "new_accuracy_after_ft",
    "forgetting",
    "cka",
    "pairwise_distance_correlation",
    "support_distance_after_ft",
    "support_inside_empirical_after_ft",
    "support_inside_ellipsoid_after_ft",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SPMA paper-readiness multiseed benchmark suite.")
    parser.add_argument("--data-dir", default="data", help="Dataset root/cache directory.")
    parser.add_argument("--output-dir", default="outputs/spma/paper_suite", help="Top-level output directory.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/spma/paper_suite", help="Top-level checkpoint directory.")
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS, help="Benchmarks to run.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Baseline methods to evaluate.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Random seeds to evaluate.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Execution device.")
    parser.add_argument("--fast-gpu", action="store_true", help="Enable mixed precision, TF32, channels-last, and other GPU-oriented speedups.")
    parser.add_argument("--base-epochs", type=int, default=20, help="Teacher training epochs.")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="Finetuning epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--anchor-batch-size", type=int, default=64, help="Anchor batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--anchor-buffer-per-class", type=int, default=64, help="Old-task anchor memory per class.")
    parser.add_argument("--anchor-eval-per-class", type=int, default=200, help="Old-task eval subset per class.")
    parser.add_argument("--visualization-per-class", type=int, default=80, help="Visualization subset per class.")
    parser.add_argument("--num-clusters", type=int, default=8, help="Support-memory cluster count.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Model hidden dimension.")
    parser.add_argument("--latent-dim", type=int, default=64, help="Model latent dimension.")
    parser.add_argument(
        "--backbone",
        default="auto",
        choices=["auto", "small_cnn", "resnet18_cifar", "resnet18_imagenet"],
        help="Backbone override.",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=float, default=8.0, help="LoRA alpha.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse completed per-seed runs if metrics already exist.")
    return parser.parse_args()


def _build_config_for_run(
    args: argparse.Namespace,
    benchmark: str,
    seed: int,
    output_dir: Path,
    checkpoint_dir: Path,
) -> ExperimentConfig:
    config = build_default_config(args.data_dir, output_dir, checkpoint_dir)
    fast_gpu_mode = bool(args.fast_gpu)
    return ExperimentConfig(
        **{
            **config.__dict__,
            "benchmark_name": benchmark,
            "seed": seed,
            "device": args.device,
            "deterministic": not fast_gpu_mode,
            "fast_gpu_mode": fast_gpu_mode,
            "enable_amp": fast_gpu_mode,
            "allow_tf32": fast_gpu_mode,
            "cudnn_benchmark": fast_gpu_mode,
            "channels_last": fast_gpu_mode,
            "persistent_workers": True,
            "prefetch_factor": 4 if fast_gpu_mode else 2,
            "base_epochs": args.base_epochs,
            "finetune_epochs": args.finetune_epochs,
            "batch_size": args.batch_size,
            "anchor_batch_size": args.anchor_batch_size,
            "num_workers": args.num_workers,
            "anchor_buffer_per_class": args.anchor_buffer_per_class,
            "anchor_eval_per_class": args.anchor_eval_per_class,
            "visualization_per_class": args.visualization_per_class,
            "num_clusters": args.num_clusters,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "backbone_name": args.backbone,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "enable_plots": False,
            "enable_ablations": False,
        }
    )


def _harmonic_mean(value_a: float, value_b: float) -> float:
    denominator = value_a + value_b
    if denominator <= 0.0:
        return 0.0
    return 2.0 * value_a * value_b / denominator


def _load_existing_metrics(output_dir: Path) -> list[dict[str, Any]] | None:
    results_path = output_dir / "all_results.json"
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("baseline_methods")


def _row_from_metrics(metrics: dict[str, Any], benchmark: str, seed: int) -> dict[str, Any]:
    row = {
        "benchmark_name": benchmark,
        "seed": seed,
        "method_name": metrics["method_name"],
        "backbone_name": metrics["backbone_name"],
        "trainable_mode": metrics["trainable_mode"],
        "classifier_mode": metrics["classifier_mode"],
        "replay_anchor_size": int(metrics["replay_anchor_size"]),
        "memory_anchor_size": int(metrics["memory_anchor_size"]),
        "support_metric": metrics["support_metric"],
        "retention_schedule": metrics["retention_schedule"],
    }
    for metric_name in AGGREGATE_METRICS:
        row[metric_name] = float(metrics[metric_name])
    row["harmonic_old_new"] = _harmonic_mean(
        float(metrics["old_accuracy_after_ft"]),
        float(metrics["new_accuracy_after_ft"]),
    )
    return row


def _format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["benchmark_name"], row["method_name"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (benchmark, method_name), group_rows in grouped.items():
        aggregate_row: dict[str, Any] = {
            "benchmark_name": benchmark,
            "method_name": method_name,
            "num_seeds": len(group_rows),
            "backbone_name": group_rows[0]["backbone_name"],
            "trainable_mode": group_rows[0]["trainable_mode"],
            "classifier_mode": group_rows[0]["classifier_mode"],
            "replay_anchor_size": int(group_rows[0]["replay_anchor_size"]),
            "memory_anchor_size": int(group_rows[0]["memory_anchor_size"]),
            "support_metric": group_rows[0]["support_metric"],
            "retention_schedule": group_rows[0]["retention_schedule"],
        }
        for metric_name in AGGREGATE_METRICS + ["harmonic_old_new"]:
            values = np.asarray([float(row[metric_name]) for row in group_rows], dtype=np.float64)
            aggregate_row[f"{metric_name}_mean"] = float(values.mean())
            aggregate_row[f"{metric_name}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            aggregate_row[metric_name] = _format_mean_std(
                aggregate_row[f"{metric_name}_mean"],
                aggregate_row[f"{metric_name}_std"],
            )
        aggregate_rows.append(aggregate_row)

    aggregate_rows.sort(
        key=lambda row: (
            row["benchmark_name"],
            row["harmonic_old_new_mean"],
            row["old_accuracy_after_ft_mean"],
            row["new_accuracy_after_ft_mean"],
        ),
        reverse=True,
    )
    return aggregate_rows


def _write_aggregate_summary(output_dir: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for row in aggregate_rows:
        by_benchmark.setdefault(str(row["benchmark_name"]), []).append(row)

    columns = [
        "method_name",
        "old_accuracy_after_ft",
        "new_accuracy_after_ft",
        "harmonic_old_new",
        "forgetting",
        "support_distance_after_ft",
        "support_inside_empirical_after_ft",
        "replay_anchor_size",
        "backbone_name",
    ]
    with open(output_dir / "aggregate_summary.md", "w", encoding="utf-8") as handle:
        handle.write("# SPMA Paper Suite Aggregate Results\n\n")
        for benchmark_name in sorted(by_benchmark):
            benchmark_rows = sorted(
                by_benchmark[benchmark_name],
                key=lambda row: (
                    row["harmonic_old_new_mean"],
                    row["old_accuracy_after_ft_mean"],
                    row["new_accuracy_after_ft_mean"],
                ),
                reverse=True,
            )
            handle.write(f"## {benchmark_name}\n\n")
            handle.write(markdown_table(benchmark_rows, columns) + "\n\n")


def run_multiseed_suite(args: argparse.Namespace) -> dict[str, Any]:
    output_root = ensure_dir(args.output_dir)
    checkpoint_root = ensure_dir(args.checkpoint_dir)

    all_rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for benchmark in args.benchmarks:
        for seed in args.seeds:
            run_output_dir = output_root / benchmark / f"seed_{seed}"
            run_checkpoint_dir = checkpoint_root / benchmark / f"seed_{seed}"
            ensure_dir(run_output_dir)
            ensure_dir(run_checkpoint_dir)
            print(f"[suite] benchmark={benchmark} seed={seed} output_dir={run_output_dir}")

            metrics_payload: list[dict[str, Any]] | None = None
            if args.reuse_existing:
                metrics_payload = _load_existing_metrics(run_output_dir)
                if metrics_payload is not None:
                    print(f"[suite] reusing existing run at {run_output_dir}")

            if metrics_payload is None:
                config = _build_config_for_run(args, benchmark, seed, run_output_dir, run_checkpoint_dir)
                result = run_experiment(config, selected_method_names=args.methods)
                metrics_payload = [baseline_result["metrics"] for baseline_result in result["baseline_results"]]

            run_manifest.append(
                {
                    "benchmark_name": benchmark,
                    "seed": seed,
                    "output_dir": str(run_output_dir),
                    "checkpoint_dir": str(run_checkpoint_dir),
                    "methods": [metrics["method_name"] for metrics in metrics_payload],
                }
            )
            all_rows.extend(_row_from_metrics(metrics, benchmark, seed) for metrics in metrics_payload)

    aggregate_rows = _aggregate_rows(all_rows)
    save_csv(output_root / "all_seed_runs.csv", all_rows)
    save_csv(output_root / "aggregate_mean_std.csv", aggregate_rows)
    save_json(
        output_root / "suite_manifest.json",
        {
            "benchmarks": args.benchmarks,
            "methods": args.methods,
            "seeds": args.seeds,
            "runs": run_manifest,
        },
    )
    _write_aggregate_summary(output_root, aggregate_rows)
    return {
        "output_dir": str(output_root),
        "checkpoint_dir": str(checkpoint_root),
        "all_seed_rows": len(all_rows),
        "aggregate_rows": len(aggregate_rows),
        "benchmarks": args.benchmarks,
        "methods": args.methods,
        "seeds": args.seeds,
    }


def main() -> None:
    args = parse_args()
    result = run_multiseed_suite(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
