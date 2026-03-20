from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .data import make_loader
from .losses import nearest_cluster_support_distance
from .manifold_memory import ManifoldMemory, TorchManifoldMemory
from .utils import autocast_context, move_to_device


@dataclass
class FeatureCache:
    features: np.ndarray
    logits: np.ndarray
    labels: np.ndarray


def evaluate_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig | None = None,
) -> float:
    model.eval()
    total_correct = 0
    total_examples = 0
    runtime_config = config if config is not None else ExperimentConfig(data_dir=".", output_dir=".", checkpoint_dir=".")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = move_to_device(inputs, device, runtime_config)
            labels = labels.to(device, non_blocking=bool(runtime_config.fast_gpu_mode))
            with autocast_context(runtime_config, device):
                predictions = model(inputs).argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.numel()
    return total_correct / max(total_examples, 1)


def evaluate_task_aware_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    config: ExperimentConfig | None = None,
) -> float:
    model.eval()
    runtime_config = config if config is not None else ExperimentConfig(data_dir=".", output_dir=".", checkpoint_dir=".")
    if not hasattr(model, "old_head") or not hasattr(model, "new_head"):
        return evaluate_accuracy(model, loader, device, runtime_config)

    if task not in {"old", "new"}:
        raise ValueError(f"Unsupported task-aware split: {task}")

    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = move_to_device(inputs, device, runtime_config)
            labels = labels.to(device, non_blocking=bool(runtime_config.fast_gpu_mode))
            with autocast_context(runtime_config, device):
                _, features = model(inputs, return_features=True)
                if task == "old":
                    local_logits = model.old_head(features) * model.old_logit_log_scale.exp() + model.old_logit_bias
                    class_indices = model.old_class_index_tensor
                else:
                    local_logits = model.new_head(features) * model.new_logit_log_scale.exp() + model.new_logit_bias
                    class_indices = model.new_class_index_tensor
                local_predictions = local_logits.argmax(dim=-1)
                predictions = class_indices[local_predictions]
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.numel()
    return total_correct / max(total_examples, 1)


def collect_features_and_logits(
    model: torch.nn.Module,
    dataset: Dataset,
    config: ExperimentConfig,
    device: torch.device,
) -> FeatureCache:
    loader = make_loader(dataset, batch_size=config.eval_batch_size, shuffle=False, config=config)
    model.eval()
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, batch_labels in loader:
            inputs = move_to_device(inputs, device, config)
            with autocast_context(config, device):
                batch_logits, batch_features = model(inputs, return_features=True)
            features.append(batch_features.detach().float().cpu().numpy())
            logits.append(batch_logits.detach().float().cpu().numpy())
            labels.append(batch_labels.detach().cpu().numpy())
    return FeatureCache(
        features=np.concatenate(features, axis=0),
        logits=np.concatenate(logits, axis=0),
        labels=np.concatenate(labels, axis=0),
    )


def linear_cka(features_x: np.ndarray, features_y: np.ndarray) -> float:
    x = features_x - features_x.mean(axis=0, keepdims=True)
    y = features_y - features_y.mean(axis=0, keepdims=True)
    cross_cov = x.T @ y
    numerator = np.square(cross_cov).sum()
    norm_x = np.sqrt(np.square(x.T @ x).sum())
    norm_y = np.sqrt(np.square(y.T @ y).sum())
    return float(numerator / max(norm_x * norm_y, 1e-12))


def pairwise_distance_correlation(features_x: np.ndarray, features_y: np.ndarray) -> float:
    x_distances = torch.cdist(torch.from_numpy(features_x), torch.from_numpy(features_x), p=2)
    y_distances = torch.cdist(torch.from_numpy(features_y), torch.from_numpy(features_y), p=2)
    upper = torch.triu_indices(x_distances.size(0), x_distances.size(1), offset=1)
    x_values = x_distances[upper[0], upper[1]].numpy()
    y_values = y_distances[upper[0], upper[1]].numpy()
    correlation = np.corrcoef(x_values, y_values)[0, 1]
    return float(correlation)


def knn_overlap_per_sample(teacher_features: np.ndarray, student_features: np.ndarray, k: int) -> np.ndarray:
    effective_k = min(k + 1, teacher_features.shape[0])
    teacher_neighbors = NearestNeighbors(n_neighbors=effective_k).fit(teacher_features).kneighbors(return_distance=False)
    student_neighbors = NearestNeighbors(n_neighbors=effective_k).fit(student_features).kneighbors(return_distance=False)
    overlaps = []
    for teacher_row, student_row in zip(teacher_neighbors, student_neighbors, strict=True):
        teacher_set = set(int(index) for index in teacher_row[1:])
        student_set = set(int(index) for index in student_row[1:])
        overlaps.append(len(teacher_set & student_set) / max(len(teacher_set), 1))
    return np.asarray(overlaps, dtype=np.float64)


def compute_support_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    memory: ManifoldMemory,
    torch_memory: TorchManifoldMemory,
    metric: str,
    confidence: float,
) -> dict[str, Any]:
    label_tensor = torch.from_numpy(labels).to(torch_memory.means.device)
    feature_tensor = torch.from_numpy(features).to(torch_memory.means.device)
    with torch.no_grad():
        scores, group_indices = nearest_cluster_support_distance(feature_tensor, torch_memory, metric=metric, labels=label_tensor)
        mahalanobis_scores, _ = nearest_cluster_support_distance(feature_tensor, torch_memory, metric="mahalanobis", labels=label_tensor)
    score_values = scores.cpu().numpy()
    group_indices_np = group_indices.cpu().numpy()
    empirical_thresholds = memory.group_thresholds[group_indices_np]
    inside_empirical = score_values <= empirical_thresholds
    ellipsoid_threshold = float(chi2.ppf(confidence, df=max(features.shape[1], 1)) / max(features.shape[1], 1))
    inside_ellipsoid = mahalanobis_scores.cpu().numpy() <= ellipsoid_threshold
    return {
        "mean_distance": float(score_values.mean()),
        "median_distance": float(np.median(score_values)),
        "p90_distance": float(np.quantile(score_values, 0.9)),
        "fraction_inside_empirical_support": float(inside_empirical.mean()),
        "fraction_inside_confidence_ellipsoid": float(inside_ellipsoid.mean()),
        "distances": score_values.tolist(),
        "group_indices": group_indices_np.tolist(),
        "ellipsoid_threshold": ellipsoid_threshold,
    }


def evaluate_method(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    datasets: dict[str, Dataset],
    memory: ManifoldMemory,
    torch_memory: TorchManifoldMemory,
    config: ExperimentConfig,
    device: torch.device,
    support_metric: str,
) -> tuple[dict[str, Any], dict[str, FeatureCache]]:
    old_test_loader = make_loader(datasets["old_test"], batch_size=config.eval_batch_size, shuffle=False, config=config)
    new_test_loader = make_loader(datasets["new_test"], batch_size=config.eval_batch_size, shuffle=False, config=config)

    old_acc_before = evaluate_accuracy(teacher, old_test_loader, device, config)
    old_acc_after = evaluate_accuracy(student, old_test_loader, device, config)
    new_acc_after = evaluate_accuracy(student, new_test_loader, device, config)
    old_acc_after_task_aware = evaluate_task_aware_accuracy(student, old_test_loader, device, task="old", config=config)
    new_acc_after_task_aware = evaluate_task_aware_accuracy(student, new_test_loader, device, task="new", config=config)

    anchor_teacher = collect_features_and_logits(teacher, datasets["anchor_eval"], config, device)
    anchor_student = collect_features_and_logits(student, datasets["anchor_eval"], config, device)
    viz_teacher_old = collect_features_and_logits(teacher, datasets["viz_old"], config, device)
    viz_teacher_new = collect_features_and_logits(teacher, datasets["viz_new"], config, device)
    viz_student_old = collect_features_and_logits(student, datasets["viz_old"], config, device)
    viz_student_new = collect_features_and_logits(student, datasets["viz_new"], config, device)

    anchor_knn_overlap = knn_overlap_per_sample(anchor_teacher.features, anchor_student.features, k=config.knn_k)
    support_before = compute_support_statistics(
        viz_teacher_new.features,
        viz_teacher_new.labels,
        memory,
        torch_memory,
        metric=support_metric,
        confidence=config.support_confidence,
    )
    support_after = compute_support_statistics(
        viz_student_new.features,
        viz_student_new.labels,
        memory,
        torch_memory,
        metric=support_metric,
        confidence=config.support_confidence,
    )
    anchor_support = compute_support_statistics(
        anchor_student.features,
        anchor_student.labels,
        memory,
        torch_memory,
        metric=support_metric,
        confidence=config.support_confidence,
    )

    metrics = {
        "old_accuracy_before_ft": old_acc_before,
        "old_accuracy_after_ft": old_acc_after,
        "new_accuracy_after_ft": new_acc_after,
        "old_accuracy_after_ft_task_aware": old_acc_after_task_aware,
        "new_accuracy_after_ft_task_aware": new_acc_after_task_aware,
        "forgetting": old_acc_before - old_acc_after,
        "cka": linear_cka(anchor_teacher.features, anchor_student.features),
        "pairwise_distance_correlation": pairwise_distance_correlation(anchor_teacher.features, anchor_student.features),
        "anchor_knn_overlap_mean": float(anchor_knn_overlap.mean()),
        "support_distance_before_ft": support_before["mean_distance"],
        "support_distance_after_ft": support_after["mean_distance"],
        "support_distance_delta": support_after["mean_distance"] - support_before["mean_distance"],
        "support_inside_empirical_before_ft": support_before["fraction_inside_empirical_support"],
        "support_inside_empirical_after_ft": support_after["fraction_inside_empirical_support"],
        "support_inside_ellipsoid_before_ft": support_before["fraction_inside_confidence_ellipsoid"],
        "support_inside_ellipsoid_after_ft": support_after["fraction_inside_confidence_ellipsoid"],
        "anchor_support_distance": anchor_support["mean_distance"],
        "support_before_details": support_before,
        "support_after_details": support_after,
        "anchor_support_details": anchor_support,
    }
    caches = {
        "anchor_teacher": anchor_teacher,
        "anchor_student": anchor_student,
        "viz_teacher_old": viz_teacher_old,
        "viz_teacher_new": viz_teacher_new,
        "viz_student_old": viz_student_old,
        "viz_student_new": viz_student_new,
    }
    return metrics, caches
