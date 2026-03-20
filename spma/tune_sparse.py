from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .config import ExperimentConfig, MethodConfig, build_default_config
from .data import build_datasets
from .finetune import finetune_with_method
from .models import build_model
from .utils import ensure_dir, markdown_table, save_csv, save_json, set_deterministic_seed, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune sparse-anchor SPMA hyperparameters against a fixed base checkpoint.")
    parser.add_argument(
        "--benchmark",
        default="split_cifar100",
        choices=["split_cifar10", "split_cifar100", "split_tiny_imagenet", "tiny_imagenet_compatible_shift"],
        help="Benchmark to tune on.",
    )
    parser.add_argument("--data-dir", default="data", help="Dataset cache directory.")
    parser.add_argument("--base-checkpoint", required=True, help="Path to a pretrained base-model checkpoint.")
    parser.add_argument("--output-dir", default="outputs/spma/tune_sparse", help="Directory to store tuning outputs.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/spma/tune_sparse", help="Directory for finetuned checkpoints.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Execution device.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed.")
    parser.add_argument("--base-epochs", type=int, default=10, help="Metadata only; used to match the base training setup.")
    parser.add_argument("--finetune-epochs", type=int, default=5, help="Finetuning epochs for each candidate.")
    parser.add_argument("--batch-size", type=int, default=256, help="New-task batch size.")
    parser.add_argument("--anchor-batch-size", type=int, default=96, help="Anchor batch size.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for the base model.")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension for the base model.")
    parser.add_argument(
        "--backbone",
        default="resnet18_cifar",
        choices=["auto", "small_cnn", "resnet18_cifar", "resnet18_imagenet"],
        help="Backbone to instantiate before loading the base checkpoint.",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512], help="Sparse replay budgets to test.")
    parser.add_argument("--lambda-anchor-ce", nargs="+", type=float, default=[0.5, 1.0, 2.0], help="Anchor CE weights to test.")
    parser.add_argument("--lambda-smooth", nargs="+", type=float, default=[0.1, 0.5], help="Smoothing weights to test.")
    parser.add_argument("--lambda-support", nargs="+", type=float, default=[0.02, 0.05], help="Support weights to test.")
    parser.add_argument("--builders", nargs="+", default=["kmeans", "hdbscan"], choices=["kmeans", "hdbscan"], help="Support-memory builders to test.")
    parser.add_argument("--lambda-kd", type=float, default=1.0, help="Fixed KD weight during the sweep.")
    parser.add_argument("--lambda-geo", type=float, default=0.25, help="Fixed geometry weight during the sweep.")
    parser.add_argument("--lambda-reg", type=float, default=0.005, help="Fixed parameter-drift weight during the sweep.")
    return parser.parse_args()


def _harmonic_mean(value_a: float, value_b: float) -> float:
    denominator = value_a + value_b
    if denominator <= 0.0:
        return 0.0
    return 2.0 * value_a * value_b / denominator


def _pareto_front(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    pareto_rows: list[dict[str, float | str]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other["method_name"] == candidate["method_name"]:
                continue
            if other["old_accuracy_after_ft"] >= candidate["old_accuracy_after_ft"] and other["new_accuracy_after_ft"] >= candidate["new_accuracy_after_ft"]:
                if other["old_accuracy_after_ft"] > candidate["old_accuracy_after_ft"] or other["new_accuracy_after_ft"] > candidate["new_accuracy_after_ft"]:
                    dominated = True
                    break
        if not dominated:
            pareto_rows.append(candidate)
    return sorted(pareto_rows, key=lambda row: (row["old_accuracy_after_ft"], row["new_accuracy_after_ft"]), reverse=True)


def _candidate_method(
    budget: int,
    lambda_anchor_ce: float,
    lambda_smooth: float,
    lambda_support: float,
    builder: str,
    args: argparse.Namespace,
) -> MethodConfig:
    suffix = f"b{budget}_ace{lambda_anchor_ce:g}_sm{lambda_smooth:g}_sup{lambda_support:g}_{builder}"
    return MethodConfig(
        name=f"sparse_tune_{suffix}",
        description="Sparse-anchor SPMA tuning candidate.",
        lambda_kd=args.lambda_kd,
        lambda_anchor_ce=lambda_anchor_ce,
        lambda_geo=args.lambda_geo,
        lambda_smooth=lambda_smooth,
        lambda_support=lambda_support,
        lambda_reg=args.lambda_reg,
        support_loss_mode="bounded_expansion",
        manifold_builder=builder,
        anchor_sampling_mode="cluster_stratified",
        anchor_total_budget_override=budget,
    )


def _row_from_metrics(metrics: dict[str, float | str]) -> dict[str, float | str]:
    row = {
        "method_name": metrics["method_name"],
        "old_accuracy_before_ft": round(metrics["old_accuracy_before_ft"], 4),
        "old_accuracy_after_ft": round(metrics["old_accuracy_after_ft"], 4),
        "new_accuracy_after_ft": round(metrics["new_accuracy_after_ft"], 4),
        "forgetting": round(metrics["forgetting"], 4),
        "cka": round(metrics["cka"], 4),
        "pairwise_distance_correlation": round(metrics["pairwise_distance_correlation"], 4),
        "support_distance_after_ft": round(metrics["support_distance_after_ft"], 4),
        "support_inside_empirical_after_ft": round(metrics["support_inside_empirical_after_ft"], 4),
        "anchor_sampling_mode": metrics["anchor_sampling_mode"],
        "replay_anchor_size": metrics["replay_anchor_size"],
        "memory_anchor_size": metrics["memory_anchor_size"],
        "lambda_kd": metrics["lambda_kd"],
        "lambda_anchor_ce": metrics["lambda_anchor_ce"],
        "lambda_geo": metrics["lambda_geo"],
        "lambda_smooth": metrics["lambda_smooth"],
        "lambda_support": metrics["lambda_support"],
        "manifold_builder": metrics["manifold_builder"],
    }
    row["harmonic_old_new"] = round(
        _harmonic_mean(float(metrics["old_accuracy_after_ft"]), float(metrics["new_accuracy_after_ft"])),
        4,
    )
    row["sum_old_new"] = round(float(metrics["old_accuracy_after_ft"]) + float(metrics["new_accuracy_after_ft"]), 4)
    return row


def _summary_markdown(
    output_dir: Path,
    rows: list[dict[str, float | str]],
    pareto_rows: list[dict[str, float | str]],
) -> None:
    columns = [
        "method_name",
        "old_accuracy_after_ft",
        "new_accuracy_after_ft",
        "harmonic_old_new",
        "sum_old_new",
        "replay_anchor_size",
        "lambda_anchor_ce",
        "lambda_smooth",
        "lambda_support",
        "manifold_builder",
    ]
    with open(output_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("# Sparse SPMA Hyperparameter Sweep\n\n")
        if rows:
            best = rows[0]
            handle.write("## Best Candidate\n\n")
            handle.write(f"- `method_name`: `{best['method_name']}`\n")
            handle.write(f"- `old_accuracy_after_ft`: `{best['old_accuracy_after_ft']}`\n")
            handle.write(f"- `new_accuracy_after_ft`: `{best['new_accuracy_after_ft']}`\n")
            handle.write(f"- `harmonic_old_new`: `{best['harmonic_old_new']}`\n")
            handle.write(f"- `replay_anchor_size`: `{best['replay_anchor_size']}`\n\n")
        handle.write("## Ranked Candidates\n\n")
        handle.write(markdown_table(rows, columns) + "\n\n")
        handle.write("## Pareto Front\n\n")
        handle.write(markdown_table(pareto_rows, columns) + "\n")


def main() -> None:
    args = parse_args()
    config = build_default_config(args.data_dir, args.output_dir, args.checkpoint_dir)
    config = ExperimentConfig(
        **{
            **config.__dict__,
            "benchmark_name": args.benchmark,
            "seed": args.seed,
            "device": args.device,
            "base_epochs": args.base_epochs,
            "finetune_epochs": args.finetune_epochs,
            "batch_size": args.batch_size,
            "anchor_batch_size": args.anchor_batch_size,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "backbone_name": args.backbone,
            "enable_plots": False,
            "enable_ablations": False,
        }
    )

    set_deterministic_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    checkpoint_dir = ensure_dir(config.checkpoint_dir)
    device = resolve_device(config.device)
    datasets = build_datasets(config)
    input_shape = tuple(datasets.metadata.get("input_shape", (1, 28, 28))) if datasets.metadata is not None else (1, 28, 28)
    if datasets.metadata is not None and "num_classes" in datasets.metadata:
        config = replace(config, num_classes=int(datasets.metadata["num_classes"]))
    if config.backbone_name == "auto" and not (input_shape[0] == 1 and input_shape[1:] == (28, 28)):
        resolved_backbone = "resnet18_imagenet" if max(input_shape[1:]) >= 64 else "resnet18_cifar"
        config = replace(config, backbone_name=resolved_backbone)

    base_model = build_model(
        config.hidden_dim,
        config.latent_dim,
        config.num_classes,
        input_shape=input_shape,
        backbone_name=config.backbone_name,
    ).to(device)
    base_checkpoint = Path(args.base_checkpoint)
    if not base_checkpoint.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")
    base_model.load_state_dict(torch.load(base_checkpoint, map_location=device))

    candidate_methods = []
    for budget in args.budgets:
        for lambda_anchor_ce in args.lambda_anchor_ce:
            for lambda_smooth in args.lambda_smooth:
                for lambda_support in args.lambda_support:
                    for builder in args.builders:
                        candidate_methods.append(
                            _candidate_method(
                                budget=budget,
                                lambda_anchor_ce=lambda_anchor_ce,
                                lambda_smooth=lambda_smooth,
                                lambda_support=lambda_support,
                                builder=builder,
                                args=args,
                            )
                        )

    results = []
    rows = []
    for method in candidate_methods:
        result = finetune_with_method(
            base_model=base_model,
            datasets=datasets,
            method=method,
            config=config,
            device=device,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        results.append(result["metrics"])
        rows.append(_row_from_metrics(result["metrics"]))

    rows.sort(key=lambda row: (row["harmonic_old_new"], row["old_accuracy_after_ft"], row["new_accuracy_after_ft"]), reverse=True)
    pareto_rows = _pareto_front(rows)
    save_csv(output_dir / "tuning_results.csv", rows)
    save_json(output_dir / "tuning_results.json", results)
    _summary_markdown(output_dir, rows, pareto_rows)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "num_candidates": len(rows),
                "best_method": rows[0]["method_name"] if rows else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
