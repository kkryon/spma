from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ExperimentConfig, build_ablation_methods, build_baseline_methods, build_default_config
from .data import build_datasets
from .finetune import finetune_with_method
from .models import build_model
from .train_base import train_base_model
from .utils import configure_runtime, ensure_dir, markdown_table, resolve_device, save_csv, save_json, set_deterministic_seed
from .visualize import plot_latent_tsne_triptych, plot_metric_bars, plot_support_distance_histograms, plot_support_pca


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SPMA continual-learning research prototype.")
    parser.add_argument(
        "--benchmark",
        default="split_mnist",
        choices=[
            "split_mnist",
            "mnist_compatible_shift",
            "mnist_stress_shift",
            "split_cifar10",
            "split_cifar100",
            "split_tiny_imagenet",
            "tiny_imagenet_compatible_shift",
            "tiny_imagenet_stress_shift",
            "cifar10_compatible_shift",
            "cifar10_stress_shift",
        ],
        help="Benchmark suite to run.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory used to cache MNIST.")
    parser.add_argument("--output-dir", default="outputs/spma/latest", help="Directory for metrics, tables, and plots.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/spma/latest", help="Directory for checkpoints.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Execution device.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed.")
    parser.add_argument("--fast-gpu", action="store_true", help="Enable mixed precision, TF32, channels-last, and other GPU-oriented speedups.")
    parser.add_argument("--base-epochs", type=int, default=6, help="Task-1 training epochs.")
    parser.add_argument("--finetune-epochs", type=int, default=5, help="Task-2 fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--anchor-batch-size", type=int, default=96, help="Anchor replay mini-batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--anchor-buffer-per-class", type=int, default=64, help="Balanced Task-1 anchor memory per class.")
    parser.add_argument("--anchor-eval-per-class", type=int, default=200, help="Balanced Task-1 evaluation subset per class.")
    parser.add_argument("--visualization-per-class", type=int, default=80, help="Samples per class used for plots.")
    parser.add_argument("--num-clusters", type=int, default=8, help="KMeans cluster count for the old manifold memory.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="First hidden-layer width.")
    parser.add_argument("--latent-dim", type=int, default=64, help="Penultimate latent width.")
    parser.add_argument(
        "--backbone",
        default="auto",
        choices=["auto", "small_cnn", "resnet18_cifar", "resnet18_imagenet"],
        help="Feature extractor backbone for non-MNIST benchmarks.",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank for adapter-style fine-tuning.")
    parser.add_argument("--lora-alpha", type=float, default=8.0, help="LoRA alpha scaling value.")
    parser.add_argument("--disable-plots", action="store_true", help="Skip visualization generation.")
    parser.add_argument("--disable-ablations", action="store_true", help="Skip SPMA ablation runs.")
    parser.add_argument("--methods", nargs="+", default=None, help="Optional subset of baseline method names.")
    return parser.parse_args()


def _row_from_metrics(metrics: dict[str, float]) -> dict[str, float | str]:
    return {
        "method_name": metrics["method_name"],
        "old_accuracy_before_ft": round(metrics["old_accuracy_before_ft"], 4),
        "old_accuracy_after_ft": round(metrics["old_accuracy_after_ft"], 4),
        "old_accuracy_after_ft_task_aware": round(metrics.get("old_accuracy_after_ft_task_aware", metrics["old_accuracy_after_ft"]), 4),
        "new_accuracy_after_ft": round(metrics["new_accuracy_after_ft"], 4),
        "new_accuracy_after_ft_task_aware": round(metrics.get("new_accuracy_after_ft_task_aware", metrics["new_accuracy_after_ft"]), 4),
        "forgetting": round(metrics["forgetting"], 4),
        "cka": round(metrics["cka"], 4),
        "pairwise_distance_correlation": round(metrics["pairwise_distance_correlation"], 4),
        "support_distance_before_ft": round(metrics["support_distance_before_ft"], 4),
        "support_distance_after_ft": round(metrics["support_distance_after_ft"], 4),
        "support_inside_empirical_after_ft": round(metrics["support_inside_empirical_after_ft"], 4),
        "support_inside_ellipsoid_after_ft": round(metrics["support_inside_ellipsoid_after_ft"], 4),
        "support_metric": metrics["support_metric"],
        "step_mix_mode": metrics["step_mix_mode"],
        "support_loss_mode": metrics["support_loss_mode"],
        "support_conditioning": metrics["support_conditioning"],
        "manifold_builder": metrics["manifold_builder"],
        "retention_schedule": metrics["retention_schedule"],
        "calibration_mode": metrics["calibration_mode"],
        "new_task_loss_mode": metrics["new_task_loss_mode"],
        "posthoc_refine_mode": metrics["posthoc_refine_mode"],
        "multilayer_support_layers": ",".join(metrics["multilayer_support_layers"]) if metrics["multilayer_support_layers"] else "",
        "trainable_mode": metrics["trainable_mode"],
        "classifier_mode": metrics["classifier_mode"],
        "backbone_name": metrics["backbone_name"],
        "anchor_sampling_mode": metrics["anchor_sampling_mode"],
        "replay_anchor_size": metrics["replay_anchor_size"],
        "memory_anchor_size": metrics["memory_anchor_size"],
    }


def _write_summary(
    output_dir: Path,
    base_history: dict[str, object],
    method_rows: list[dict[str, float | str]],
    ablation_rows: list[dict[str, float | str]],
    benchmark_name: str,
) -> None:
    baseline_columns = [
        "method_name",
        "old_accuracy_before_ft",
        "old_accuracy_after_ft",
        "old_accuracy_after_ft_task_aware",
        "new_accuracy_after_ft",
        "new_accuracy_after_ft_task_aware",
        "forgetting",
        "cka",
        "pairwise_distance_correlation",
        "support_distance_before_ft",
        "support_distance_after_ft",
        "support_inside_empirical_after_ft",
        "anchor_sampling_mode",
        "replay_anchor_size",
        "memory_anchor_size",
        "step_mix_mode",
        "support_loss_mode",
        "support_conditioning",
        "manifold_builder",
        "retention_schedule",
        "calibration_mode",
        "new_task_loss_mode",
        "posthoc_refine_mode",
        "multilayer_support_layers",
        "trainable_mode",
        "classifier_mode",
        "backbone_name",
    ]
    ablation_columns = [
        "method_name",
        "old_accuracy_after_ft",
        "new_accuracy_after_ft",
        "forgetting",
        "support_distance_after_ft",
        "support_inside_empirical_after_ft",
        "anchor_sampling_mode",
        "replay_anchor_size",
        "support_metric",
        "step_mix_mode",
        "support_loss_mode",
        "support_conditioning",
        "manifold_builder",
        "retention_schedule",
        "calibration_mode",
        "new_task_loss_mode",
        "posthoc_refine_mode",
        "multilayer_support_layers",
        "trainable_mode",
        "classifier_mode",
        "backbone_name",
    ]
    baseline_table = markdown_table(method_rows, baseline_columns)
    ablation_table = markdown_table(ablation_rows, ablation_columns)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write(f"# SPMA Summary: {benchmark_name}\n\n")
        handle.write("## Base Model\n\n")
        handle.write(f"- Best old-task accuracy before finetuning: {base_history['best_old_test_accuracy']:.4f}\n")
        handle.write(f"- Teacher checkpoint: `{base_history['checkpoint_path']}`\n")
        handle.write(f"- Saved old-task latents: `{base_history['embeddings_path']}`\n\n")
        handle.write("## Baselines\n\n")
        handle.write(baseline_table + "\n\n")
        if ablation_rows:
            handle.write("## Ablations\n\n")
            handle.write(ablation_table + "\n")


def _render_discussion(method_rows: list[dict[str, float | str]], ablation_rows: list[dict[str, float | str]]) -> str:
    by_name = {row["method_name"]: row for row in method_rows}
    spma = by_name.get("spma_novel") or by_name.get("spma_full")
    plain = by_name.get("plain_ft")
    hidden = by_name.get("hidden_l2")
    if spma is None or plain is None or hidden is None:
        return ""

    lines = [
        "# SPMA Discussion",
        "",
        "## When SPMA works",
        "",
        f"- Relative to plain fine-tuning, SPMA changed old-task accuracy from `{plain['old_accuracy_after_ft']}` to `{spma['old_accuracy_after_ft']}` while keeping new-task accuracy at `{spma['new_accuracy_after_ft']}`.",
        f"- Relative to plain fine-tuning, SPMA reduced average support distance from `{plain['support_distance_after_ft']}` to `{spma['support_distance_after_ft']}`.",
        f"- Relative to hidden-state L2, SPMA achieved CKA `{spma['cka']}` vs `{hidden['cka']}` and pairwise-distance correlation `{spma['pairwise_distance_correlation']}` vs `{hidden['pairwise_distance_correlation']}`.",
        "",
        "## Failure Modes Checked",
        "",
    ]
    ablations = {row["method_name"]: row for row in ablation_rows}
    if "spma_high_lambda_support" in ablations:
        lines.append(
            f"- Support loss too strong: `spma_high_lambda_support` reached new-task accuracy `{ablations['spma_high_lambda_support']['new_accuracy_after_ft']}`."
        )
    if "spma_low_lambda_support" in ablations:
        lines.append(
            f"- Support loss too weak: `spma_low_lambda_support` ended with support distance `{ablations['spma_low_lambda_support']['support_distance_after_ft']}`."
        )
    if "spma_high_lambda_geo" in ablations:
        lines.append(
            f"- Geometry loss too strong: `spma_high_lambda_geo` reached new-task accuracy `{ablations['spma_high_lambda_geo']['new_accuracy_after_ft']}`."
        )
    lines.extend(
        [
            "- Hidden-state L2 baseline is included explicitly as `hidden_l2` and can preserve coordinates more rigidly than pairwise geometry.",
            "- Bad clustering/support estimation can be probed with `spma_k4`, `spma_k12`, and `spma_euclidean_support`.",
            "- If the new task truly needs new directions, the strong-support ablations should degrade new-task accuracy first.",
            "",
        ]
    )
    return "\n".join(lines)


def _filter_methods(methods, selected_names: list[str] | None):
    if selected_names is None:
        return methods
    selected = set(selected_names)
    return [method for method in methods if method.name in selected]


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    config = build_default_config(args.data_dir, args.output_dir, args.checkpoint_dir)
    fast_gpu_mode = bool(args.fast_gpu)
    return ExperimentConfig(
        **{
            **config.__dict__,
            "benchmark_name": args.benchmark,
            "seed": args.seed,
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
            "enable_plots": not args.disable_plots,
            "enable_ablations": not args.disable_ablations,
        }
    )


def _resolve_model_config(config: ExperimentConfig, datasets) -> tuple[ExperimentConfig, tuple[int, ...]]:
    input_shape = tuple(datasets.metadata.get("input_shape", (1, 28, 28))) if datasets.metadata is not None else (1, 28, 28)
    if datasets.metadata is not None and "num_classes" in datasets.metadata:
        config = ExperimentConfig(**{**config.__dict__, "num_classes": int(datasets.metadata["num_classes"])})
    if config.backbone_name == "auto" and not (input_shape[0] == 1 and input_shape[1:] == (28, 28)):
        resolved_backbone = "resnet18_imagenet" if max(input_shape[1:]) >= 64 else "resnet18_cifar"
        config = ExperimentConfig(**{**config.__dict__, "backbone_name": resolved_backbone})
    return config, input_shape


def run_experiment(
    config: ExperimentConfig,
    selected_method_names: list[str] | None = None,
) -> dict[str, object]:
    set_deterministic_seed(config.seed, deterministic=config.deterministic)
    output_dir = ensure_dir(config.output_dir)
    checkpoint_dir = ensure_dir(config.checkpoint_dir)
    device = resolve_device(config.device)
    configure_runtime(config, device)
    print(f"[setup] device={device} seed={config.seed}")

    datasets = build_datasets(config)
    config, input_shape = _resolve_model_config(config, datasets)
    base_model = build_model(
        config.hidden_dim,
        config.latent_dim,
        config.num_classes,
        input_shape=input_shape,
        backbone_name=config.backbone_name,
    ).to(device)
    base_history = train_base_model(
        base_model,
        datasets,
        config,
        device,
        checkpoint_path=checkpoint_dir / "base_model.pt",
        embeddings_path=output_dir / "base_old_task_latents.npz",
    )
    save_json(output_dir / "base_history.json", base_history)

    baseline_methods = _filter_methods(build_baseline_methods(config), selected_method_names)
    baseline_results: list[dict[str, object]] = []
    baseline_rows: list[dict[str, float | str]] = []
    shared_cache: dict[str, object] = {}
    for method in baseline_methods:
        result = finetune_with_method(
            base_model=base_model,
            datasets=datasets,
            method=method,
            config=config,
            device=device,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            shared_cache=shared_cache,
        )
        baseline_results.append(result)
        baseline_rows.append(_row_from_metrics(result["metrics"]))

    ablation_results: list[dict[str, object]] = []
    ablation_rows: list[dict[str, float | str]] = []
    if config.enable_ablations:
        for method in build_ablation_methods(config):
            result = finetune_with_method(
                base_model=base_model,
                datasets=datasets,
                method=method,
                config=config,
                device=device,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                shared_cache=shared_cache,
            )
            ablation_results.append(result)
            ablation_rows.append(_row_from_metrics(result["metrics"]))

    save_csv(output_dir / "baseline_results.csv", baseline_rows)
    save_csv(output_dir / "ablation_results.csv", ablation_rows)
    save_json(
        output_dir / "all_results.json",
        {
            "config": config,
            "base_history": base_history,
            "dataset_metadata": datasets.metadata,
            "baseline_methods": [result["metrics"] for result in baseline_results],
            "ablations": [result["metrics"] for result in ablation_results],
        },
    )
    _write_summary(output_dir, base_history, baseline_rows, ablation_rows, config.benchmark_name)

    discussion = _render_discussion(baseline_rows, ablation_rows)
    if discussion:
        with open(output_dir / "discussion.md", "w", encoding="utf-8") as handle:
            handle.write(discussion + "\n")

    if config.enable_plots:
        metrics_by_name = {result["metrics"]["method_name"]: result for result in baseline_results}
        spma_plot_name = "spma_novel" if "spma_novel" in metrics_by_name else "spma_full"
        if {"plain_ft", spma_plot_name}.issubset(metrics_by_name.keys()):
            teacher_caches = metrics_by_name["plain_ft"]["feature_caches"]
            plain_caches = metrics_by_name["plain_ft"]["feature_caches"]
            spma_caches = metrics_by_name[spma_plot_name]["feature_caches"]
            memory = metrics_by_name[spma_plot_name]["memory"]
            plot_latent_tsne_triptych(
                teacher_caches=teacher_caches,
                plain_caches=plain_caches,
                spma_caches=spma_caches,
                memory=memory,
                config=config,
                save_path=output_dir / "latent_tsne_triptych.png",
            )
            plot_support_pca(
                teacher_caches=teacher_caches,
                plain_caches=plain_caches,
                spma_caches=spma_caches,
                memory=memory,
                save_path=output_dir / "support_pca.png",
            )
            plot_support_distance_histograms(
                plain_metrics=metrics_by_name["plain_ft"]["metrics"],
                spma_metrics=metrics_by_name[spma_plot_name]["metrics"],
                save_path=output_dir / "support_distance_histograms.png",
            )
        plot_metric_bars(baseline_rows, save_path=output_dir / "baseline_metric_bars.png")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "baseline_methods": [row["method_name"] for row in baseline_rows],
            },
            indent=2,
        )
    )
    return {
        "config": config,
        "base_history": base_history,
        "dataset_metadata": datasets.metadata,
        "baseline_results": baseline_results,
        "baseline_rows": baseline_rows,
        "ablation_results": ablation_results,
        "ablation_rows": ablation_rows,
    }


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    run_experiment(config, selected_method_names=args.methods)


if __name__ == "__main__":
    main()
