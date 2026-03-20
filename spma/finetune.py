from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, Dataset, Subset

from .config import ExperimentConfig, MethodConfig
from .data import DatasetBundle, make_loader
from .evaluate import evaluate_accuracy, evaluate_method
from .losses import (
    cross_entropy_loss,
    distillation_kl,
    hidden_feature_l2,
    local_chart_continuation_loss,
    local_manifold_smoothing_loss,
    old_logit_suppression_loss,
    parameter_drift_penalty,
    relational_geometry_loss,
    soft_chart_assignment_loss,
    support_loss,
)
from .manifold_memory import (
    build_manifold_memory,
    build_manifold_memory_from_latents,
    build_multilayer_manifold_memories,
    build_multilayer_manifold_memories_from_latents,
    collect_latents,
    collect_layer_latents,
    layer_projection_to_torch,
    manifold_memory_to_torch,
    project_features_torch,
)
from .models import count_trainable_parameters, enable_lora_finetuning, forward_with_intermediates, wrap_with_dual_head
from .utils import (
    autocast_context,
    build_grad_scaler,
    clone_state_dict,
    dataset_labels,
    freeze_model,
    maybe_channels_last_model,
    move_to_device,
    save_json,
    select_balanced_indices,
)


def _build_anchor_dataset_with_override(
    datasets: DatasetBundle,
    method: MethodConfig,
    config: ExperimentConfig,
) -> tuple[Dataset, ExperimentConfig]:
    adjusted_config = config
    if method.num_clusters_override is not None:
        adjusted_config = replace(adjusted_config, num_clusters=method.num_clusters_override)

    if method.anchor_buffer_per_class_override is None:
        return datasets.anchor_memory, adjusted_config

    old_train_eval = datasets.old_train_eval
    old_labels = dataset_labels(old_train_eval)
    indices = select_balanced_indices(old_labels, adjusted_config.old_classes, method.anchor_buffer_per_class_override, adjusted_config.seed + 101)
    anchor_dataset = Subset(old_train_eval, indices.tolist())
    adjusted_config = replace(adjusted_config, anchor_buffer_per_class=method.anchor_buffer_per_class_override)
    return anchor_dataset, adjusted_config


def _anchor_cache_key(
    datasets: DatasetBundle,
    method: MethodConfig,
    config: ExperimentConfig,
    anchor_dataset: Dataset,
) -> tuple[Any, ...]:
    metadata = datasets.metadata or {}
    old_classes = tuple(int(value) for value in metadata.get("old_classes", config.old_classes))
    anchor_source = (
        "anchor_memory"
        if method.anchor_buffer_per_class_override is None
        else f"old_train_eval_balanced_{method.anchor_buffer_per_class_override}"
    )
    return (
        str(metadata.get("benchmark_name", config.benchmark_name)),
        config.seed,
        anchor_source,
        tuple(old_classes),
        len(anchor_dataset),
    )


def _memory_signature(method: MethodConfig, config: ExperimentConfig) -> tuple[Any, ...]:
    return (
        method.support_metric,
        method.support_conditioning,
        method.manifold_builder,
        config.num_clusters,
        config.cluster_pca_rank,
        config.covariance_eps,
        config.gmm_reg_covar,
        config.hdbscan_min_cluster_size,
        config.hdbscan_min_samples,
        config.hdbscan_cluster_selection_epsilon,
        config.hdbscan_allow_single_cluster,
        config.support_distance_quantile,
        config.seed,
    )


def _get_cached_anchor_latents(
    teacher: torch.nn.Module,
    anchor_dataset: Dataset,
    datasets: DatasetBundle,
    method: MethodConfig,
    config: ExperimentConfig,
    device: torch.device,
    shared_cache: dict[str, Any] | None,
) -> tuple[tuple[Any, ...] | None, np.ndarray, np.ndarray]:
    if not config.cache_teacher_features or shared_cache is None:
        anchor_features, anchor_labels = collect_latents(teacher, anchor_dataset, config.eval_batch_size, config, device)
        return None, anchor_features, anchor_labels

    feature_cache = shared_cache.setdefault("anchor_latents", {})
    cache_key = _anchor_cache_key(datasets, method, config, anchor_dataset)
    cached_value = feature_cache.get(cache_key)
    if cached_value is None:
        cached_value = collect_latents(teacher, anchor_dataset, config.eval_batch_size, config, device)
        feature_cache[cache_key] = cached_value
    anchor_features, anchor_labels = cached_value
    return cache_key, anchor_features, anchor_labels


def _get_cached_manifold_memory(
    teacher: torch.nn.Module,
    anchor_dataset: Dataset,
    datasets: DatasetBundle,
    method: MethodConfig,
    config: ExperimentConfig,
    device: torch.device,
    shared_cache: dict[str, Any] | None,
):
    anchor_cache_key, anchor_features, anchor_labels = _get_cached_anchor_latents(
        teacher,
        anchor_dataset,
        datasets,
        method,
        config,
        device,
        shared_cache,
    )
    if not config.cache_teacher_features or shared_cache is None:
        return build_manifold_memory_from_latents(anchor_features, anchor_labels, method, config)

    memory_cache = shared_cache.setdefault("manifold_memories", {})
    cache_key = (anchor_cache_key, _memory_signature(method, config))
    cached_value = memory_cache.get(cache_key)
    if cached_value is None:
        cached_value = build_manifold_memory_from_latents(anchor_features, anchor_labels, method, config)
        memory_cache[cache_key] = cached_value
    return cached_value


def _get_cached_multilayer_memories(
    teacher: torch.nn.Module,
    anchor_dataset: Dataset,
    datasets: DatasetBundle,
    method: MethodConfig,
    config: ExperimentConfig,
    device: torch.device,
    shared_cache: dict[str, Any] | None,
):
    layer_names = tuple(method.multilayer_support_layers)
    if not layer_names:
        return {}, {}, {}

    anchor_cache_key = _anchor_cache_key(datasets, method, config, anchor_dataset)
    if not config.cache_teacher_features or shared_cache is None:
        return build_multilayer_manifold_memories(teacher, anchor_dataset, method, config, device)

    layer_feature_cache = shared_cache.setdefault("multilayer_anchor_latents", {})
    feature_key = (anchor_cache_key, layer_names, config.multilayer_projection_dim)
    cached_features = layer_feature_cache.get(feature_key)
    if cached_features is None:
        cached_features = collect_layer_latents(
            teacher,
            anchor_dataset,
            layer_names=layer_names,
            batch_size=config.eval_batch_size,
            config=config,
            device=device,
        )
        layer_feature_cache[feature_key] = cached_features
    layer_features, anchor_labels = cached_features

    memory_cache = shared_cache.setdefault("multilayer_manifold_memories", {})
    memory_key = (feature_key, _memory_signature(method, config))
    cached_value = memory_cache.get(memory_key)
    if cached_value is None:
        cached_value = build_multilayer_manifold_memories_from_latents(layer_features, anchor_labels, method, config)
        memory_cache[memory_key] = cached_value
    return cached_value


def _build_student_model(
    base_model: torch.nn.Module,
    datasets: DatasetBundle,
    method: MethodConfig,
    config: ExperimentConfig,
) -> torch.nn.Module:
    student = copy.deepcopy(base_model)
    task_relation = str((datasets.metadata or {}).get("task_relation", ""))
    classifier_mode = method.classifier_mode
    if classifier_mode == "auto":
        classifier_mode = "dual_head" if task_relation == "disjoint_classes" else "single"

    if method.trainable_mode == "lora":
        lora_rank = method.lora_rank_override if method.lora_rank_override is not None else config.lora_rank
        lora_alpha = method.lora_alpha_override if method.lora_alpha_override is not None else config.lora_alpha
        student = enable_lora_finetuning(student, rank=lora_rank, alpha=lora_alpha, keep_classifier_trainable=True)
    elif method.trainable_mode != "full":
        raise ValueError(f"Unsupported trainable mode: {method.trainable_mode}")

    if classifier_mode == "dual_head":
        metadata = datasets.metadata or {}
        old_classes = tuple(int(value) for value in metadata.get("old_classes", config.old_classes))
        new_classes = tuple(int(value) for value in metadata.get("new_classes", config.new_classes))
        student = wrap_with_dual_head(
            student,
            old_classes=old_classes,
            new_classes=new_classes,
            reinitialize_new_head=True,
            new_head_bias_init=config.dual_head_new_head_bias_init,
        )
    elif classifier_mode != "single":
        raise ValueError(f"Unsupported classifier mode: {classifier_mode}")

    setattr(student, "resolved_classifier_mode", classifier_mode)
    return student


def _apply_trainable_schedule(
    student: torch.nn.Module,
    reference_trainability: dict[str, bool],
    warmup_active: bool,
) -> None:
    for name, parameter in student.named_parameters():
        if warmup_active:
            parameter.requires_grad_(name.startswith("new_head."))
        else:
            parameter.requires_grad_(reference_trainability.get(name, parameter.requires_grad))


def _set_batchnorm_eval(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def _stabilize_batchnorm_stats(student: torch.nn.Module) -> None:
    student.apply(_set_batchnorm_eval)


def _remap_new_labels(student: torch.nn.Module, labels: torch.Tensor) -> torch.Tensor:
    if not hasattr(student, "new_class_index_tensor"):
        raise ValueError("New-label remapping requires a dual-head student.")
    class_indices = student.new_class_index_tensor.to(labels.device)
    remapped = torch.empty_like(labels)
    for remapped_index, class_index in enumerate(class_indices.tolist()):
        remapped[labels == class_index] = remapped_index
    return remapped


def _build_optimizer(
    student: torch.nn.Module,
    config: ExperimentConfig,
) -> AdamW:
    classifier_mode = getattr(student, "resolved_classifier_mode", "single")
    if classifier_mode != "dual_head":
        return AdamW(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            lr=config.finetune_learning_rate,
            weight_decay=config.weight_decay,
        )

    parameter_groups: list[dict[str, object]] = []
    feature_parameters = []
    old_head_parameters = []
    new_head_parameters = []
    residual_parameters = []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("feature_model."):
            feature_parameters.append(parameter)
        elif name.startswith("old_head."):
            old_head_parameters.append(parameter)
        elif name.startswith("new_head."):
            new_head_parameters.append(parameter)
        else:
            residual_parameters.append(parameter)

    if feature_parameters:
        parameter_groups.append(
            {
                "params": feature_parameters,
                "lr": config.finetune_learning_rate * config.dual_head_feature_lr_scale,
            }
        )
    if old_head_parameters:
        parameter_groups.append(
            {
                "params": old_head_parameters,
                "lr": config.finetune_learning_rate * config.dual_head_old_head_lr_scale,
            }
        )
    if new_head_parameters:
        parameter_groups.append(
            {
                "params": new_head_parameters,
                "lr": config.finetune_learning_rate * config.dual_head_new_head_lr_scale,
            }
        )
    if residual_parameters:
        parameter_groups.append(
            {
                "params": residual_parameters,
                "lr": config.finetune_learning_rate,
            }
        )
    return AdamW(parameter_groups, weight_decay=config.weight_decay)


def _next_anchor_batch(anchor_iterator, anchor_loader):
    try:
        return next(anchor_iterator), anchor_iterator
    except StopIteration:
        anchor_iterator = iter(anchor_loader)
        return next(anchor_iterator), anchor_iterator


def _allocate_cluster_budget(counts: torch.Tensor, total_budget: int) -> torch.Tensor:
    if total_budget >= int(counts.sum().item()):
        return counts.clone()

    positive_mask = counts > 0
    num_positive = int(positive_mask.sum().item())
    if num_positive == 0:
        return torch.zeros_like(counts)

    allocation = torch.zeros_like(counts)
    if total_budget >= num_positive:
        allocation[positive_mask] = 1
        remaining = total_budget - num_positive
    else:
        positive_indices = torch.nonzero(positive_mask, as_tuple=False).flatten()[:total_budget]
        allocation[positive_indices] = 1
        return allocation

    fractional = counts.float() / counts.sum().clamp_min(1)
    proportional = fractional * remaining
    addition = torch.floor(proportional).to(torch.long)
    addition = torch.minimum(addition, counts - allocation)
    allocation += addition
    remaining_after_floor = total_budget - int(allocation.sum().item())
    if remaining_after_floor <= 0:
        return allocation

    residual = proportional - torch.floor(proportional)
    candidate_indices = torch.argsort(residual, descending=True)
    for index in candidate_indices.tolist():
        if remaining_after_floor <= 0:
            break
        if allocation[index] >= counts[index]:
            continue
        allocation[index] += 1
        remaining_after_floor -= 1
    return allocation


def _build_sparse_anchor_dataset(
    anchor_dataset: Dataset,
    memory,
    memory_cache: dict[str, Any],
    method: MethodConfig,
    config: ExperimentConfig,
) -> Dataset:
    if method.anchor_sampling_mode != "cluster_stratified" or method.anchor_total_budget_override is None:
        return anchor_dataset

    anchor_labels = dataset_labels(anchor_dataset)
    total_budget = min(method.anchor_total_budget_override, anchor_labels.size(0))
    assignments = torch.from_numpy(memory_cache["assignments"]).to(torch.long)
    features = torch.from_numpy(memory_cache["anchor_features"]).float()
    means = torch.from_numpy(memory.means).float()
    counts = torch.bincount(assignments, minlength=means.size(0))
    allocation = _allocate_cluster_budget(counts, total_budget)
    selected_indices: list[torch.Tensor] = []

    for cluster_index in range(means.size(0)):
        take = int(allocation[cluster_index].item())
        if take <= 0:
            continue
        cluster_indices = torch.nonzero(assignments == cluster_index, as_tuple=False).flatten()
        if cluster_indices.numel() == 0:
            continue
        cluster_features = features[cluster_indices]
        cluster_mean = means[cluster_index].unsqueeze(0)
        distances = torch.cdist(cluster_features, cluster_mean).squeeze(1)
        nearest_order = torch.argsort(distances, descending=False)
        selected_indices.append(cluster_indices[nearest_order[:take]])

    if not selected_indices:
        return anchor_dataset

    merged_indices = torch.cat(selected_indices, dim=0)
    merged_indices = merged_indices.unique(sorted=True)
    return Subset(anchor_dataset, merged_indices.tolist())


def _multilayer_support_loss(
    layer_features: dict[str, torch.Tensor],
    layer_memories: dict[str, Any],
    layer_projections: dict[str, Any],
    layer_weights: dict[str, float],
    method: MethodConfig,
    method_config: ExperimentConfig,
    labels: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not layer_memories:
        device = next(iter(layer_features.values())).device
        return torch.tensor(0.0, device=device), {}

    weighted_losses: list[torch.Tensor] = []
    per_layer_losses: dict[str, float] = {}
    total_weight = 0.0
    for layer_name, raw_features in layer_features.items():
        if layer_name not in layer_memories:
            continue
        weight = float(layer_weights.get(layer_name, 1.0))
        if weight <= 0.0:
            continue
        projected_features = project_features_torch(raw_features, layer_projections[layer_name])
        loss_value, _, _ = support_loss(
            projected_features,
            layer_memories[layer_name],
            metric=method.support_metric,
            labels=labels if method.support_conditioning == "class_conditional" else None,
            loss_mode=method.support_loss_mode,
            expansion_scale=method_config.support_expansion_scale,
            novelty_gate_center=method_config.support_novelty_gate_center,
            novelty_temperature=method_config.support_novelty_temperature,
        )
        weighted_losses.append(loss_value * weight)
        per_layer_losses[layer_name] = float(loss_value.detach().item())
        total_weight += weight

    if not weighted_losses:
        device = next(iter(layer_features.values())).device
        return torch.tensor(0.0, device=device), per_layer_losses
    return sum(weighted_losses) / max(total_weight, 1e-6), per_layer_losses


def _scheduled_scale(schedule: str, epoch: int, start_epoch: int, end_epoch: int, final_scale: float) -> float:
    if schedule == "constant" or final_scale == 1.0:
        return 1.0
    if schedule != "linear_decay":
        raise ValueError(f"Unsupported schedule: {schedule}")
    if epoch <= start_epoch:
        return 1.0
    if end_epoch <= start_epoch:
        return final_scale
    progress = min(max((epoch - start_epoch) / max(end_epoch - start_epoch, 1), 0.0), 1.0)
    return 1.0 + (final_scale - 1.0) * progress


def _balanced_subset_from_dataset(
    dataset: Dataset,
    classes: tuple[int, ...],
    total_budget: int,
    seed: int,
) -> Dataset:
    labels = dataset_labels(dataset)
    if total_budget >= labels.numel():
        return dataset

    per_class = max((total_budget + len(classes) - 1) // max(len(classes), 1), 1)
    selected = select_balanced_indices(labels, classes, per_class, seed)
    if selected.numel() > total_budget:
        generator = torch.Generator().manual_seed(seed + 17)
        permutation = torch.randperm(selected.numel(), generator=generator)
        selected = selected[permutation[:total_budget]]
    selected = selected.unique(sorted=True)
    return Subset(dataset, selected.tolist())


def _build_calibration_dataset(
    datasets: DatasetBundle,
    replay_anchor_dataset: Dataset,
    method: MethodConfig,
    config: ExperimentConfig,
) -> tuple[Dataset | None, int, int]:
    if method.calibration_mode == "none":
        return None, 0, 0

    metadata = datasets.metadata or {}
    old_classes = tuple(int(value) for value in metadata.get("old_classes", config.old_classes))
    new_classes = tuple(int(value) for value in metadata.get("new_classes", config.new_classes))
    old_budget = min(method.calibration_old_budget or len(replay_anchor_dataset), len(replay_anchor_dataset))
    new_budget = min(method.calibration_new_budget or len(datasets.new_train), len(datasets.new_train))
    if old_budget <= 0 or new_budget <= 0:
        return None, 0, 0

    calibration_old = _balanced_subset_from_dataset(replay_anchor_dataset, old_classes, old_budget, config.seed + 211)
    calibration_new = _balanced_subset_from_dataset(datasets.new_train, new_classes, new_budget, config.seed + 223)
    calibration_dataset = ConcatDataset([calibration_old, calibration_new])
    return calibration_dataset, len(calibration_old), len(calibration_new)


def _run_dual_head_calibration(
    student: torch.nn.Module,
    calibration_dataset: Dataset | None,
    method: MethodConfig,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    if calibration_dataset is None or method.calibration_mode == "none":
        return {}
    if getattr(student, "resolved_classifier_mode", "single") != "dual_head":
        return {}
    if method.calibration_mode != "balanced_affine":
        raise ValueError(f"Unsupported calibration mode: {method.calibration_mode}")

    parameter_names = {
        "old_logit_log_scale",
        "old_logit_bias",
        "new_logit_log_scale",
        "new_logit_bias",
    }
    trainable_parameters = []
    previous_flags: dict[str, bool] = {}
    for name, parameter in student.named_parameters():
        previous_flags[name] = parameter.requires_grad
        trainable = name in parameter_names
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_parameters.append(parameter)

    if not trainable_parameters:
        return {}

    optimizer = AdamW(trainable_parameters, lr=method.calibration_learning_rate, weight_decay=0.0)
    scaler = build_grad_scaler(config, device)
    calibration_loader = make_loader(
        calibration_dataset,
        batch_size=min(config.eval_batch_size, len(calibration_dataset)),
        shuffle=True,
        config=config,
    )

    for _ in range(method.calibration_epochs):
        student.train()
        _stabilize_batchnorm_stats(student)
        for inputs, labels in calibration_loader:
            inputs = move_to_device(inputs, device, config)
            labels = labels.to(device, non_blocking=bool(config.fast_gpu_mode))
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config, device):
                logits = student(inputs)
                loss = cross_entropy_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    for name, parameter in student.named_parameters():
        parameter.requires_grad_(previous_flags.get(name, parameter.requires_grad))

    return {
        "old_logit_scale": float(student.old_logit_log_scale.exp().item()),
        "old_logit_bias": float(student.old_logit_bias.item()),
        "new_logit_scale": float(student.new_logit_log_scale.exp().item()),
        "new_logit_bias": float(student.new_logit_bias.item()),
        "calibration_dataset_size": float(len(calibration_dataset)),
    }


def _run_balanced_head_refinement(
    student: torch.nn.Module,
    replay_anchor_dataset: Dataset,
    datasets: DatasetBundle,
    method: MethodConfig,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    if method.posthoc_refine_mode == "none":
        return {}
    if getattr(student, "resolved_classifier_mode", "single") != "dual_head":
        return {}
    if method.posthoc_refine_mode != "balanced_heads":
        raise ValueError(f"Unsupported posthoc refine mode: {method.posthoc_refine_mode}")

    metadata = datasets.metadata or {}
    old_classes = tuple(int(value) for value in metadata.get("old_classes", config.old_classes))
    new_classes = tuple(int(value) for value in metadata.get("new_classes", config.new_classes))
    old_budget = min(method.posthoc_refine_old_budget or len(replay_anchor_dataset), len(replay_anchor_dataset))
    new_budget = min(method.posthoc_refine_new_budget or len(datasets.new_train), len(datasets.new_train))
    if old_budget <= 0 or new_budget <= 0 or method.posthoc_refine_epochs <= 0:
        return {}

    refine_old = _balanced_subset_from_dataset(replay_anchor_dataset, old_classes, old_budget, config.seed + 307)
    refine_new = _balanced_subset_from_dataset(datasets.new_train, new_classes, new_budget, config.seed + 331)
    refinement_dataset = ConcatDataset([refine_old, refine_new])

    previous_flags: dict[str, bool] = {}
    for name, parameter in student.named_parameters():
        previous_flags[name] = parameter.requires_grad
        parameter.requires_grad_(name.startswith("old_head.") or name.startswith("new_head."))

    optimizer = AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=method.posthoc_refine_learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = build_grad_scaler(config, device)
    refinement_loader = make_loader(
        refinement_dataset,
        batch_size=min(config.batch_size, len(refinement_dataset)),
        shuffle=True,
        config=config,
    )

    for _ in range(method.posthoc_refine_epochs):
        student.train()
        _stabilize_batchnorm_stats(student)
        for inputs, labels in refinement_loader:
            inputs = move_to_device(inputs, device, config)
            labels = labels.to(device, non_blocking=bool(config.fast_gpu_mode))
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config, device):
                logits = student(inputs)
                loss = cross_entropy_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    for name, parameter in student.named_parameters():
        parameter.requires_grad_(previous_flags.get(name, parameter.requires_grad))

    return {
        "posthoc_refine_dataset_size": float(len(refinement_dataset)),
        "posthoc_refine_old_size": float(len(refine_old)),
        "posthoc_refine_new_size": float(len(refine_new)),
    }


def finetune_with_method(
    base_model: torch.nn.Module,
    datasets: DatasetBundle,
    method: MethodConfig,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    shared_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    method_anchor_dataset, method_config = _build_anchor_dataset_with_override(datasets, method, config)
    teacher = freeze_model(maybe_channels_last_model(copy.deepcopy(base_model).to(device), method_config))
    student = maybe_channels_last_model(_build_student_model(base_model, datasets, method, method_config).to(device), method_config)
    reference_state = clone_state_dict(student)
    reference_trainability = {name: parameter.requires_grad for name, parameter in student.named_parameters()}
    exempt_prefixes = tuple(getattr(student, "parameter_drift_exempt_prefixes", ()))
    memory, memory_cache = _get_cached_manifold_memory(
        teacher,
        method_anchor_dataset,
        datasets,
        method,
        method_config,
        device,
        shared_cache,
    )
    torch_memory = manifold_memory_to_torch(memory, device)
    multilayer_memories, multilayer_caches, multilayer_projections = _get_cached_multilayer_memories(
        teacher,
        method_anchor_dataset,
        datasets,
        method,
        method_config,
        device,
        shared_cache,
    )
    multilayer_torch_memories = {
        layer_name: manifold_memory_to_torch(layer_memory, device)
        for layer_name, layer_memory in multilayer_memories.items()
    }
    multilayer_torch_projections = {
        layer_name: layer_projection_to_torch(layer_projection, device)
        for layer_name, layer_projection in multilayer_projections.items()
    }
    if method.multilayer_support_layers:
        if len(method.multilayer_support_weights) == len(method.multilayer_support_layers):
            multilayer_weights = dict(zip(method.multilayer_support_layers, method.multilayer_support_weights, strict=True))
        else:
            multilayer_weights = {layer_name: 1.0 for layer_name in method.multilayer_support_layers}
    else:
        multilayer_weights = {}
    replay_anchor_dataset = _build_sparse_anchor_dataset(method_anchor_dataset, memory, memory_cache, method, method_config)
    calibration_dataset, calibration_old_size, calibration_new_size = _build_calibration_dataset(
        datasets,
        replay_anchor_dataset,
        method,
        method_config,
    )

    new_loader = make_loader(datasets.new_train, batch_size=method_config.batch_size, shuffle=True, config=method_config)
    anchor_loader = make_loader(replay_anchor_dataset, batch_size=method_config.anchor_batch_size, shuffle=True, config=method_config)
    old_test_loader = make_loader(datasets.old_test, batch_size=method_config.eval_batch_size, shuffle=False, config=method_config)
    new_test_loader = make_loader(datasets.new_test, batch_size=method_config.eval_batch_size, shuffle=False, config=method_config)
    initial_old_accuracy = evaluate_accuracy(student, old_test_loader, device, method_config)
    initial_new_accuracy = evaluate_accuracy(student, new_test_loader, device, method_config)
    anchor_iterator = iter(anchor_loader)
    previous_warmup_state: bool | None = None
    mix_generator = torch.Generator(device="cpu").manual_seed(method_config.seed + 911)
    scaler = build_grad_scaler(method_config, device)

    history: list[dict[str, float]] = []
    for epoch in range(1, method_config.finetune_epochs + 1):
        warmup_active = bool(
            getattr(student, "resolved_classifier_mode", "single") == "dual_head"
            and epoch <= method_config.dual_head_warmup_epochs
        )
        warmup_end_epoch = method_config.dual_head_warmup_epochs if getattr(student, "resolved_classifier_mode", "single") == "dual_head" else 0
        retention_scale = _scheduled_scale(
            method.retention_schedule,
            epoch,
            start_epoch=warmup_end_epoch + 1,
            end_epoch=method_config.finetune_epochs,
            final_scale=method.retention_final_scale,
        )
        anchor_ce_scale = _scheduled_scale(
            method.retention_schedule,
            epoch,
            start_epoch=warmup_end_epoch + 1,
            end_epoch=method_config.finetune_epochs,
            final_scale=method.anchor_ce_final_scale,
        )
        if warmup_active != previous_warmup_state:
            _apply_trainable_schedule(student, reference_trainability, warmup_active)
            optimizer = _build_optimizer(student, method_config)
            previous_warmup_state = warmup_active

        student.train()
        _stabilize_batchnorm_stats(student)
        totals = {
            "loss_total": 0.0,
            "loss_new": 0.0,
            "loss_old_suppression": 0.0,
            "loss_kd": 0.0,
            "loss_anchor_ce": 0.0,
            "loss_hidden": 0.0,
            "loss_chart": 0.0,
            "loss_continue": 0.0,
            "loss_geo": 0.0,
            "loss_smooth": 0.0,
            "loss_support": 0.0,
            "loss_support_multilayer": 0.0,
            "loss_reg": 0.0,
        }
        total_examples = 0

        for new_inputs, new_labels in new_loader:
            (anchor_inputs, anchor_labels), anchor_iterator = _next_anchor_batch(anchor_iterator, anchor_loader)
            new_inputs = move_to_device(new_inputs, device, method_config)
            new_labels = new_labels.to(device, non_blocking=bool(method_config.fast_gpu_mode))
            anchor_inputs = move_to_device(anchor_inputs, device, method_config)
            anchor_labels = anchor_labels.to(device, non_blocking=bool(method_config.fast_gpu_mode))
            batch_size = new_labels.size(0)
            per_layer_support: dict[str, float] = {}

            ce_step_active = True
            kd_step_active = method.lambda_kd > 0.0
            if method.step_mix_mode == "ce_kd_stepmix":
                ce_probability = float(method.ce_step_ratio)
                ce_step_active = bool(torch.rand(1, generator=mix_generator).item() < ce_probability)
                kd_step_active = not ce_step_active and method.lambda_kd > 0.0
            elif method.step_mix_mode != "joint":
                raise ValueError(f"Unsupported step_mix_mode: {method.step_mix_mode}")

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(method_config, device):
                if method.multilayer_support_layers:
                    new_logits, new_features, new_intermediates = forward_with_intermediates(
                        student,
                        new_inputs,
                        layer_names=method.multilayer_support_layers,
                    )
                else:
                    new_logits, new_features = student(new_inputs, return_features=True)
                    new_intermediates = {}
            new_logits = new_logits.float()
            new_features = new_features.float()
            new_intermediates = {layer_name: values.float() for layer_name, values in new_intermediates.items()}
            if getattr(student, "resolved_classifier_mode", "single") == "dual_head":
                remapped_new_labels = _remap_new_labels(student, new_labels)
            else:
                remapped_new_labels = new_labels

            if not ce_step_active and method.step_mix_mode == "ce_kd_stepmix":
                loss_new = torch.tensor(0.0, device=device)
            elif warmup_active and getattr(student, "resolved_classifier_mode", "single") == "dual_head":
                warmup_logits = student.new_head(new_features)
                loss_new = cross_entropy_loss(warmup_logits, remapped_new_labels)
            elif method.new_task_loss_mode == "new_head_ce" and getattr(student, "resolved_classifier_mode", "single") == "dual_head":
                loss_new = cross_entropy_loss(student.new_head(new_features), remapped_new_labels)
            else:
                loss_new = cross_entropy_loss(new_logits, new_labels)
            total_loss = loss_new

            loss_old_suppression = torch.tensor(0.0, device=device)
            if (
                method.lambda_old_suppression > 0.0
                and ce_step_active
                and getattr(student, "resolved_classifier_mode", "single") == "dual_head"
            ):
                new_head_logits = student.new_head(new_features)
                old_head_logits = student.old_head(new_features)
                assembled_logits = new_features.new_full((new_features.size(0), student.num_classes), -1e4)
                assembled_logits[:, student.old_class_index_tensor] = old_head_logits
                assembled_logits[:, student.new_class_index_tensor] = new_head_logits
                loss_old_suppression = old_logit_suppression_loss(
                    old_head_logits,
                    assembled_logits,
                    new_labels,
                    margin=method.old_suppression_margin,
                )
                total_loss = total_loss + method.lambda_old_suppression * loss_old_suppression

            with torch.no_grad():
                with autocast_context(method_config, device):
                    teacher_anchor_logits, teacher_anchor_features = teacher(anchor_inputs, return_features=True)
            teacher_anchor_logits = teacher_anchor_logits.float()
            teacher_anchor_features = teacher_anchor_features.float()

            with autocast_context(method_config, device):
                student_anchor_logits, student_anchor_features = student(anchor_inputs, return_features=True)
            student_anchor_logits = student_anchor_logits.float()
            student_anchor_features = student_anchor_features.float()

            loss_kd = torch.tensor(0.0, device=device)
            if kd_step_active and retention_scale > 0.0:
                loss_kd = distillation_kl(student_anchor_logits, teacher_anchor_logits, temperature=method.temperature)
                total_loss = total_loss + (method.lambda_kd * retention_scale) * loss_kd

            loss_anchor_ce = torch.tensor(0.0, device=device)
            if method.lambda_anchor_ce > 0.0 and anchor_ce_scale > 0.0 and ce_step_active:
                loss_anchor_ce = cross_entropy_loss(student_anchor_logits, anchor_labels)
                total_loss = total_loss + (method.lambda_anchor_ce * anchor_ce_scale) * loss_anchor_ce

            loss_hidden = torch.tensor(0.0, device=device)
            if method.lambda_hidden > 0.0 and retention_scale > 0.0 and ce_step_active:
                loss_hidden = hidden_feature_l2(student_anchor_features, teacher_anchor_features)
                total_loss = total_loss + (method.lambda_hidden * retention_scale) * loss_hidden

            loss_chart = torch.tensor(0.0, device=device)
            if method.lambda_chart > 0.0 and retention_scale > 0.0 and ce_step_active:
                loss_chart = soft_chart_assignment_loss(
                    student_anchor_features,
                    teacher_anchor_features,
                    torch_memory,
                    metric=method.support_metric,
                    temperature=method.chart_temperature,
                )
                total_loss = total_loss + (method.lambda_chart * retention_scale) * loss_chart

            loss_geo = torch.tensor(0.0, device=device)
            if method.lambda_geo > 0.0 and retention_scale > 0.0 and ce_step_active and student_anchor_features.size(0) > 1:
                loss_geo = relational_geometry_loss(student_anchor_features, teacher_anchor_features)
                total_loss = total_loss + (method.lambda_geo * retention_scale) * loss_geo

            loss_smooth = torch.tensor(0.0, device=device)
            if method.lambda_smooth > 0.0 and retention_scale > 0.0 and ce_step_active and student_anchor_features.size(0) > 2:
                loss_smooth = local_manifold_smoothing_loss(
                    student_anchor_features,
                    teacher_anchor_features,
                    knn_k=method_config.smooth_knn_k,
                    affinity_temperature=method_config.smooth_affinity_temperature,
                )
                total_loss = total_loss + (method.lambda_smooth * retention_scale) * loss_smooth

            loss_continue = torch.tensor(0.0, device=device)
            if method.lambda_continue > 0.0 and retention_scale > 0.0 and ce_step_active:
                loss_continue, _ = local_chart_continuation_loss(
                    new_features,
                    torch_memory,
                    metric=method.support_metric,
                    tangent_expansion_scale=method.continuation_tangent_scale,
                    residual_expansion_scale=method.continuation_residual_scale,
                    tangent_weight=method.continuation_tangent_weight,
                )
                total_loss = total_loss + (method.lambda_continue * retention_scale) * loss_continue

            loss_support = torch.tensor(0.0, device=device)
            if method.lambda_support > 0.0 and retention_scale > 0.0 and ce_step_active:
                if method.multilayer_support_layers:
                    loss_support, per_layer_support = _multilayer_support_loss(
                        new_intermediates,
                        multilayer_torch_memories,
                        multilayer_torch_projections,
                        multilayer_weights,
                        method,
                        method_config,
                        new_labels,
                    )
                else:
                    support_labels = new_labels if method.support_conditioning == "class_conditional" else None
                    loss_support, _, _ = support_loss(
                        new_features,
                        torch_memory,
                        metric=method.support_metric,
                        labels=support_labels,
                        loss_mode=method.support_loss_mode,
                        expansion_scale=method_config.support_expansion_scale,
                        novelty_gate_center=method_config.support_novelty_gate_center,
                        novelty_temperature=method_config.support_novelty_temperature,
                    )
                total_loss = total_loss + (method.lambda_support * retention_scale) * loss_support

            loss_reg = torch.tensor(0.0, device=device)
            if method.lambda_reg > 0.0 and retention_scale > 0.0 and (ce_step_active or kd_step_active):
                loss_reg = parameter_drift_penalty(student, reference_state, exempt_prefixes=exempt_prefixes)
                total_loss = total_loss + (method.lambda_reg * retention_scale) * loss_reg

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            totals["loss_total"] += total_loss.item() * batch_size
            totals["loss_new"] += loss_new.item() * batch_size
            totals["loss_old_suppression"] += loss_old_suppression.item() * batch_size
            totals["loss_kd"] += loss_kd.item() * batch_size
            totals["loss_anchor_ce"] += loss_anchor_ce.item() * batch_size
            totals["loss_hidden"] += loss_hidden.item() * batch_size
            totals["loss_chart"] += loss_chart.item() * batch_size
            totals["loss_continue"] += loss_continue.item() * batch_size
            totals["loss_geo"] += loss_geo.item() * batch_size
            totals["loss_smooth"] += loss_smooth.item() * batch_size
            totals["loss_support"] += loss_support.item() * batch_size
            if method.multilayer_support_layers:
                totals["loss_support_multilayer"] += sum(per_layer_support.values()) * batch_size
            totals["loss_reg"] += loss_reg.item() * batch_size
            total_examples += batch_size

        epoch_metrics = {key: value / max(total_examples, 1) for key, value in totals.items()}
        epoch_metrics["old_test_accuracy"] = evaluate_accuracy(student, old_test_loader, device, method_config)
        epoch_metrics["new_test_accuracy"] = evaluate_accuracy(student, new_test_loader, device, method_config)
        epoch_metrics["retention_scale"] = retention_scale
        epoch_metrics["anchor_ce_scale"] = anchor_ce_scale
        epoch_metrics["epoch"] = float(epoch)
        history.append(epoch_metrics)
        print(
            f"[{method.name}] epoch={epoch:02d} total={epoch_metrics['loss_total']:.4f} "
            f"new={epoch_metrics['loss_new']:.4f} kd={epoch_metrics['loss_kd']:.4f} "
            f"old_sup={epoch_metrics['loss_old_suppression']:.4f} "
            f"anchor_ce={epoch_metrics['loss_anchor_ce']:.4f} "
            f"hidden={epoch_metrics['loss_hidden']:.4f} chart={epoch_metrics['loss_chart']:.4f} "
            f"cont={epoch_metrics['loss_continue']:.4f} geo={epoch_metrics['loss_geo']:.4f} "
            f"smooth={epoch_metrics['loss_smooth']:.4f} "
            f"support={epoch_metrics['loss_support']:.4f} support_ml={epoch_metrics['loss_support_multilayer']:.4f} "
            f"reg={epoch_metrics['loss_reg']:.4f} "
            f"ret_scale={retention_scale:.3f} ace_scale={anchor_ce_scale:.3f} "
            f"old_acc={epoch_metrics['old_test_accuracy']:.4f} new_acc={epoch_metrics['new_test_accuracy']:.4f}"
        )

    refinement_metrics = _run_balanced_head_refinement(student, replay_anchor_dataset, datasets, method, method_config, device)
    calibration_metrics = _run_dual_head_calibration(student, calibration_dataset, method, method_config, device)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{method.name}.pt"
    torch.save(student.state_dict(), checkpoint_path)

    metrics, caches = evaluate_method(
        teacher=teacher,
        student=student,
        datasets={
            "old_test": datasets.old_test,
            "new_test": datasets.new_test,
            "anchor_eval": datasets.anchor_eval,
            "viz_old": datasets.viz_old,
            "viz_new": datasets.viz_new,
        },
        memory=memory,
        torch_memory=torch_memory,
        config=method_config,
        device=device,
        support_metric=method.support_metric,
    )
    metrics.update(
        {
            "method_name": method.name,
            "description": method.description,
            "lambda_kd": method.lambda_kd,
            "lambda_anchor_ce": method.lambda_anchor_ce,
            "lambda_geo": method.lambda_geo,
            "lambda_smooth": method.lambda_smooth,
            "lambda_support": method.lambda_support,
            "lambda_reg": method.lambda_reg,
            "lambda_hidden": method.lambda_hidden,
            "lambda_chart": method.lambda_chart,
            "lambda_continue": method.lambda_continue,
            "lambda_old_suppression": method.lambda_old_suppression,
            "new_task_loss_mode": method.new_task_loss_mode,
            "old_suppression_margin": method.old_suppression_margin,
            "chart_temperature": method.chart_temperature,
            "continuation_tangent_scale": method.continuation_tangent_scale,
            "continuation_residual_scale": method.continuation_residual_scale,
            "continuation_tangent_weight": method.continuation_tangent_weight,
            "support_metric": method.support_metric,
            "step_mix_mode": method.step_mix_mode,
            "ce_step_ratio": method.ce_step_ratio,
            "support_loss_mode": method.support_loss_mode,
            "support_conditioning": method.support_conditioning,
            "manifold_builder": method.manifold_builder,
            "retention_schedule": method.retention_schedule,
            "retention_final_scale": method.retention_final_scale,
            "anchor_ce_final_scale": method.anchor_ce_final_scale,
            "calibration_mode": method.calibration_mode,
            "posthoc_refine_mode": method.posthoc_refine_mode,
            "trainable_mode": method.trainable_mode,
            "classifier_mode": getattr(student, "resolved_classifier_mode", method.classifier_mode),
            "backbone_name": method_config.backbone_name,
            "student_old_accuracy_before_ft": initial_old_accuracy,
            "student_new_accuracy_before_ft": initial_new_accuracy,
            "anchor_sampling_mode": method.anchor_sampling_mode,
            "multilayer_support_layers": list(method.multilayer_support_layers),
            "replay_anchor_size": len(replay_anchor_dataset),
            "memory_anchor_size": len(method_anchor_dataset),
            "multilayer_memory_layers": list(multilayer_memories.keys()),
            "calibration_old_size": calibration_old_size,
            "calibration_new_size": calibration_new_size,
            "num_clusters": method_config.num_clusters,
            "anchor_buffer_per_class": method_config.anchor_buffer_per_class,
            "checkpoint_path": str(checkpoint_path),
            "history": history,
            "memory_cache": memory_cache,
            "trainable_parameters": count_trainable_parameters(student),
        }
    )
    metrics.update(refinement_metrics)
    metrics.update(calibration_metrics)
    metrics["teacher_old_accuracy_before_ft"] = metrics["old_accuracy_before_ft"]
    metrics["old_accuracy_before_ft"] = initial_old_accuracy
    metrics["new_accuracy_before_ft"] = initial_new_accuracy
    metrics["forgetting"] = metrics["old_accuracy_before_ft"] - metrics["old_accuracy_after_ft"]
    save_json(Path(output_dir) / f"{method.name}_metrics.json", metrics)

    return {
        "model": student,
        "teacher": teacher,
        "metrics": metrics,
        "feature_caches": caches,
        "memory": memory,
    }
