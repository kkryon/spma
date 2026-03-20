from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture

from .config import ExperimentConfig, MethodConfig
from .data import make_loader
from .models import forward_with_intermediates
from .utils import autocast_context, move_to_device


@dataclass(frozen=True)
class ManifoldMemory:
    means: np.ndarray
    covariances: np.ndarray
    inverse_covariances: np.ndarray
    log_determinants: np.ndarray
    mixture_weights: np.ndarray
    counts: np.ndarray
    pca_bases: np.ndarray
    factor_eigenvalues: np.ndarray
    factor_residual_variances: np.ndarray
    factor_ranks: np.ndarray
    component_tangent_thresholds: np.ndarray
    component_tangent_medians: np.ndarray
    component_residual_thresholds: np.ndarray
    component_residual_medians: np.ndarray
    component_group_ids: np.ndarray
    component_class_ids: np.ndarray
    group_labels: np.ndarray
    group_thresholds: np.ndarray
    group_score_medians: np.ndarray
    global_threshold: float
    global_score_median: float
    assignment_histogram: np.ndarray
    support_metric: str
    manifold_builder: str
    conditioning: str


@dataclass(frozen=True)
class TorchManifoldMemory:
    means: torch.Tensor
    inverse_covariances: torch.Tensor
    log_determinants: torch.Tensor
    mixture_weights: torch.Tensor
    pca_bases: torch.Tensor
    factor_eigenvalues: torch.Tensor
    factor_residual_variances: torch.Tensor
    factor_ranks: torch.Tensor
    component_tangent_thresholds: torch.Tensor
    component_tangent_medians: torch.Tensor
    component_residual_thresholds: torch.Tensor
    component_residual_medians: torch.Tensor
    component_group_ids: torch.Tensor
    group_labels: torch.Tensor
    group_thresholds: torch.Tensor
    group_score_medians: torch.Tensor
    global_threshold: torch.Tensor
    support_metric: str
    conditioning: str


@dataclass(frozen=True)
class LayerProjection:
    mean: np.ndarray
    basis: np.ndarray


@dataclass(frozen=True)
class TorchLayerProjection:
    mean: torch.Tensor
    basis: torch.Tensor


def collect_latents(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, config=config)
    model.eval()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, batch_labels in loader:
            inputs = move_to_device(inputs, device, config)
            with autocast_context(config, device):
                _, batch_features = model(inputs, return_features=True)
            features.append(batch_features.detach().float().cpu().numpy())
            labels.append(batch_labels.detach().cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def collect_layer_latents(
    model: torch.nn.Module,
    dataset,
    layer_names: Sequence[str],
    batch_size: int,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, config=config)
    model.eval()
    features_by_layer = {layer_name: [] for layer_name in layer_names}
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, batch_labels in loader:
            inputs = move_to_device(inputs, device, config)
            with autocast_context(config, device):
                _, _, batch_features = forward_with_intermediates(model, inputs, layer_names=layer_names)
            for layer_name in layer_names:
                features_by_layer[layer_name].append(batch_features[layer_name].detach().float().cpu().numpy())
            labels.append(batch_labels.detach().cpu().numpy())
    return {layer_name: np.concatenate(chunks, axis=0) for layer_name, chunks in features_by_layer.items()}, np.concatenate(labels, axis=0)


def _fit_projection(features: np.ndarray, projection_dim: int) -> LayerProjection:
    feature_dim = features.shape[1]
    centered_mean = features.mean(axis=0, keepdims=True)
    centered = features - centered_mean
    if feature_dim <= projection_dim or features.shape[0] <= 1:
        basis = np.eye(feature_dim, dtype=np.float32)
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(projection_dim, vt.shape[0])
        basis = vt[:rank].T.astype(np.float32)
    return LayerProjection(mean=centered_mean.astype(np.float32).reshape(-1), basis=basis)


def project_features_numpy(features: np.ndarray, projection: LayerProjection) -> np.ndarray:
    centered = features - projection.mean[None, :]
    return (centered @ projection.basis).astype(np.float32)


def project_features_torch(features: torch.Tensor, projection: TorchLayerProjection) -> torch.Tensor:
    centered = features - projection.mean.unsqueeze(0)
    return centered @ projection.basis


def _covariance_with_shrinkage(features: np.ndarray, covariance_eps: float) -> np.ndarray:
    dim = features.shape[1]
    if features.shape[0] <= 1:
        return np.eye(dim, dtype=np.float32) * covariance_eps
    centered = features - features.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(features.shape[0] - 1, 1)
    average_variance = float(np.trace(covariance) / max(dim, 1))
    covariance = covariance + np.eye(dim, dtype=np.float32) * (covariance_eps + 1e-3 * average_variance)
    return covariance.astype(np.float32)


def _fit_cluster_pca(features: np.ndarray, max_rank: int) -> np.ndarray:
    if features.shape[0] <= 1:
        return np.zeros((features.shape[1], 0), dtype=np.float32)
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(max_rank, vt.shape[0])
    return vt[:rank].T.astype(np.float32)


def _fit_cluster_factor_model(
    features: np.ndarray,
    max_rank: int,
    covariance_eps: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    dim = features.shape[1]
    if features.shape[0] <= 1:
        return np.zeros((dim, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), float(covariance_eps), 0

    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    covariance_eigenvalues = (singular_values**2) / max(features.shape[0] - 1, 1)
    rank = int(min(max_rank, covariance_eigenvalues.shape[0]))
    basis = vt[:rank].T.astype(np.float32)
    retained = covariance_eigenvalues[:rank].astype(np.float32)
    discarded = covariance_eigenvalues[rank:]
    residual_variance = float(discarded.mean()) if discarded.size > 0 else float(covariance_eps)
    residual_variance = max(residual_variance, covariance_eps)
    return basis, retained, residual_variance, rank


def _mahalanobis_distance_numpy(features: np.ndarray, means: np.ndarray, inverse_covariances: np.ndarray) -> np.ndarray:
    deltas = features[:, None, :] - means[None, :, :]
    projected = np.einsum("bkd,kde->bke", deltas, inverse_covariances)
    return np.einsum("bke,bke->bk", projected, deltas) / max(features.shape[1], 1)


def _euclidean_distance_numpy(features: np.ndarray, means: np.ndarray) -> np.ndarray:
    deltas = features[:, None, :] - means[None, :, :]
    return np.mean(deltas**2, axis=-1)


def _gmm_nll_numpy(
    features: np.ndarray,
    means: np.ndarray,
    inverse_covariances: np.ndarray,
    log_determinants: np.ndarray,
    mixture_weights: np.ndarray,
) -> np.ndarray:
    deltas = features[:, None, :] - means[None, :, :]
    projected = np.einsum("bkd,kde->bke", deltas, inverse_covariances)
    mahalanobis = np.einsum("bke,bke->bk", projected, deltas)
    log_weights = np.log(np.clip(mixture_weights, 1e-12, None))
    dim = features.shape[1]
    log_prob = -0.5 * (dim * np.log(2.0 * np.pi) + log_determinants[None, :] + mahalanobis)
    weighted = log_prob + log_weights[None, :]
    return (-np.logaddexp.reduce(weighted, axis=1) / max(dim, 1)).astype(np.float32)


def _factor_nll_numpy(
    features: np.ndarray,
    means: np.ndarray,
    bases: np.ndarray,
    eigenvalues: np.ndarray,
    residual_variances: np.ndarray,
    ranks: np.ndarray,
) -> np.ndarray:
    deltas = features[:, None, :] - means[None, :, :]
    basis_count = bases.shape[-1]
    if basis_count > 0:
        coefficients = np.einsum("bkd,kdr->bkr", deltas, bases)
        reconstructed = np.einsum("bkr,kdr->bkd", coefficients, bases)
        residuals = deltas - reconstructed
    else:
        coefficients = np.zeros((features.shape[0], means.shape[0], 0), dtype=np.float32)
        residuals = deltas

    variances = eigenvalues[None, :, :] + residual_variances[None, :, None]
    if basis_count > 0:
        active_mask = np.arange(basis_count)[None, :] < ranks[:, None]
        active_mask = active_mask[None, :, :]
        factor_term = np.where(active_mask, coefficients**2 / np.clip(variances, 1e-6, None), 0.0).sum(axis=-1)
        logdet_factors = np.where(active_mask[0], np.log(np.clip(variances[0], 1e-6, None)), 0.0).sum(axis=-1)
    else:
        factor_term = np.zeros((features.shape[0], means.shape[0]), dtype=np.float32)
        logdet_factors = np.zeros((means.shape[0],), dtype=np.float32)

    residual_norm = np.sum(residuals**2, axis=-1)
    residual_term = residual_norm / np.clip(residual_variances[None, :], 1e-6, None)
    dim = features.shape[1]
    logdet_residual = (dim - ranks.astype(np.float32)) * np.log(np.clip(residual_variances, 1e-6, None))
    return (
        0.5
        * (
            factor_term
            + residual_term
            + logdet_factors[None, :]
            + logdet_residual[None, :]
            + dim * np.log(2.0 * np.pi)
        )
        / max(dim, 1)
    ).astype(np.float32)


def _component_factor_statistics(
    features: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    residual_variance: float,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    centered = features - mean[None, :]
    if rank > 0 and basis.shape[1] > 0:
        active_basis = basis[:, :rank]
        coefficients = centered @ active_basis
        variances = np.clip(eigenvalues[:rank] + residual_variance, 1e-6, None)
        tangent_scores = np.sum((coefficients**2) / variances[None, :], axis=1) / max(rank, 1)
        reconstructed = coefficients @ active_basis.T
        residuals = centered - reconstructed
    else:
        tangent_scores = np.zeros((features.shape[0],), dtype=np.float32)
        residuals = centered
    residual_scores = np.mean(residuals**2, axis=1) / max(float(residual_variance), 1e-6)
    return tangent_scores.astype(np.float32), residual_scores.astype(np.float32)


def _group_scores_numpy(
    features: np.ndarray,
    means: np.ndarray,
    inverse_covariances: np.ndarray,
    log_determinants: np.ndarray,
    mixture_weights: np.ndarray,
    pca_bases: np.ndarray,
    factor_eigenvalues: np.ndarray,
    factor_residual_variances: np.ndarray,
    factor_ranks: np.ndarray,
    support_metric: str,
) -> np.ndarray:
    if support_metric == "euclidean":
        return _euclidean_distance_numpy(features, means).min(axis=1).astype(np.float32)
    if support_metric == "mahalanobis":
        return _mahalanobis_distance_numpy(features, means, inverse_covariances).min(axis=1).astype(np.float32)
    if support_metric == "gmm_nll":
        return _gmm_nll_numpy(features, means, inverse_covariances, log_determinants, mixture_weights)
    if support_metric == "factor_nll":
        return _factor_nll_numpy(
            features,
            means,
            pca_bases,
            factor_eigenvalues,
            factor_residual_variances,
            factor_ranks,
        ).min(axis=1).astype(np.float32)
    raise ValueError(f"Unsupported support metric: {support_metric}")


def _fit_group_components(
    features: np.ndarray,
    builder: str,
    n_components: int,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if builder == "kmeans":
        kmeans = KMeans(n_clusters=n_components, random_state=config.seed, n_init=10)
        assignments = kmeans.fit_predict(features)
        means = kmeans.cluster_centers_.astype(np.float32)
        counts = np.bincount(assignments, minlength=n_components).astype(np.float32)
        weights = counts / max(counts.sum(), 1.0)
        covariances = []
        for component_index in range(n_components):
            component_features = features[assignments == component_index]
            covariances.append(_covariance_with_shrinkage(component_features, config.covariance_eps))
        return means, np.stack(covariances, axis=0), weights.astype(np.float32), assignments.astype(np.int64)
    if builder == "hdbscan":
        if features.shape[0] < 2:
            return _fit_group_components(features, "kmeans", max(1, n_components), config)
        clusterer = HDBSCAN(
            min_cluster_size=min(config.hdbscan_min_cluster_size, features.shape[0]),
            min_samples=config.hdbscan_min_samples,
            cluster_selection_epsilon=config.hdbscan_cluster_selection_epsilon,
            allow_single_cluster=config.hdbscan_allow_single_cluster,
        )
        raw_assignments = clusterer.fit_predict(features)
        discovered_labels = [int(label) for label in sorted(np.unique(raw_assignments).tolist()) if label >= 0]
        if not discovered_labels:
            return _fit_group_components(features, "kmeans", max(1, n_components), config)
        label_to_index = {label: index for index, label in enumerate(discovered_labels)}
        assignments = np.asarray([label_to_index.get(int(label), -1) for label in raw_assignments], dtype=np.int64)
        provisional_means = np.stack([features[assignments == index].mean(axis=0) for index in range(len(discovered_labels))], axis=0).astype(np.float32)
        noise_mask = assignments < 0
        if np.any(noise_mask):
            noise_distances = _euclidean_distance_numpy(features[noise_mask], provisional_means)
            assignments[noise_mask] = noise_distances.argmin(axis=1)
        component_count = len(discovered_labels)
        means = np.stack([features[assignments == index].mean(axis=0) for index in range(component_count)], axis=0).astype(np.float32)
        counts = np.bincount(assignments, minlength=component_count).astype(np.float32)
        weights = counts / max(counts.sum(), 1.0)
        covariances = []
        for component_index in range(component_count):
            component_features = features[assignments == component_index]
            covariances.append(_covariance_with_shrinkage(component_features, config.covariance_eps))
        return means, np.stack(covariances, axis=0), weights.astype(np.float32), assignments.astype(np.int64)
    if builder == "gmm":
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=config.gmm_reg_covar,
            random_state=config.seed,
        )
        gmm.fit(features)
        assignments = gmm.predict(features)
        return (
            gmm.means_.astype(np.float32),
            gmm.covariances_.astype(np.float32),
            gmm.weights_.astype(np.float32),
            assignments.astype(np.int64),
        )
    raise ValueError(f"Unsupported manifold builder: {builder}")


def _group_labels(anchor_labels: np.ndarray, conditioning: str) -> list[int]:
    if conditioning == "global":
        return [-1]
    if conditioning == "class_conditional":
        return [-1] + [int(label) for label in sorted(np.unique(anchor_labels).tolist())]
    raise ValueError(f"Unsupported support conditioning: {conditioning}")


def _build_memory_from_features(
    anchor_features: np.ndarray,
    anchor_labels: np.ndarray,
    method: MethodConfig,
    config: ExperimentConfig,
) -> tuple[ManifoldMemory, dict[str, np.ndarray]]:
    group_labels = _group_labels(anchor_labels, method.support_conditioning)
    component_means: list[np.ndarray] = []
    component_covariances: list[np.ndarray] = []
    component_inverse_covariances: list[np.ndarray] = []
    component_log_determinants: list[float] = []
    component_weights: list[float] = []
    component_counts: list[int] = []
    component_group_ids: list[int] = []
    component_class_ids: list[int] = []
    pca_bases: list[np.ndarray] = []
    factor_eigenvalues: list[np.ndarray] = []
    factor_residual_variances: list[float] = []
    factor_ranks: list[int] = []
    component_tangent_thresholds: list[float] = []
    component_tangent_medians: list[float] = []
    component_residual_thresholds: list[float] = []
    component_residual_medians: list[float] = []
    assignment_histogram: list[int] = []
    group_thresholds: list[float] = []
    group_score_medians: list[float] = []
    global_old_scores = np.zeros(anchor_features.shape[0], dtype=np.float32)
    global_assignments = np.zeros(anchor_features.shape[0], dtype=np.int64)

    component_offset = 0
    for group_index, group_label in enumerate(group_labels):
        if group_label == -1:
            group_mask = np.ones(anchor_labels.shape[0], dtype=bool)
        else:
            group_mask = anchor_labels == group_label

        group_features = anchor_features[group_mask]
        n_components = min(config.num_clusters, group_features.shape[0])
        if n_components < 1:
            continue

        means, covariances, weights, local_assignments = _fit_group_components(
            group_features,
            method.manifold_builder,
            n_components,
            config,
        )
        n_components = int(means.shape[0])

        inverse_covariances = []
        log_determinants = []
        local_pca_bases = []
        local_factor_eigenvalues = []
        local_factor_residuals = []
        local_factor_ranks = []
        local_counts = np.bincount(local_assignments, minlength=n_components)
        for component_index in range(n_components):
            inverse_covariance = np.linalg.pinv(covariances[component_index]).astype(np.float32)
            inverse_covariances.append(inverse_covariance)
            sign, logdet = np.linalg.slogdet(covariances[component_index])
            log_determinants.append(float(logdet if sign > 0 else 0.0))
            component_features = group_features[local_assignments == component_index]
            basis, retained_eigenvalues, residual_variance, rank = _fit_cluster_factor_model(
                component_features,
                config.cluster_pca_rank,
                config.covariance_eps,
            )
            local_pca_bases.append(basis)
            local_factor_eigenvalues.append(retained_eigenvalues)
            local_factor_residuals.append(residual_variance)
            local_factor_ranks.append(rank)
            tangent_scores, residual_scores = _component_factor_statistics(
                component_features,
                means[component_index],
                basis,
                retained_eigenvalues,
                residual_variance,
                rank,
            )
            component_tangent_thresholds.append(float(np.quantile(tangent_scores, config.support_distance_quantile)) if tangent_scores.size > 0 else 0.0)
            component_tangent_medians.append(float(np.median(tangent_scores)) if tangent_scores.size > 0 else 0.0)
            component_residual_thresholds.append(float(np.quantile(residual_scores, config.support_distance_quantile)) if residual_scores.size > 0 else 0.0)
            component_residual_medians.append(float(np.median(residual_scores)) if residual_scores.size > 0 else 0.0)

        inverse_covariances_np = np.stack(inverse_covariances, axis=0)
        log_determinants_np = np.asarray(log_determinants, dtype=np.float32)
        max_local_basis_rank = max((basis.shape[1] for basis in local_pca_bases), default=0)
        padded_local_bases = []
        for basis in local_pca_bases:
            if basis.shape[1] == max_local_basis_rank:
                padded_local_bases.append(basis.astype(np.float32))
            else:
                padded_local_bases.append(np.pad(basis.astype(np.float32), ((0, 0), (0, max_local_basis_rank - basis.shape[1])), mode="constant"))
        max_local_rank = max((values.shape[0] for values in local_factor_eigenvalues), default=0)
        padded_local_eigenvalues = []
        for values in local_factor_eigenvalues:
            if values.shape[0] == max_local_rank:
                padded_local_eigenvalues.append(values.astype(np.float32))
            else:
                padded_local_eigenvalues.append(np.pad(values.astype(np.float32), (0, max_local_rank - values.shape[0]), mode="constant"))
        padded_local_eigenvalues_np = (
            np.stack(padded_local_eigenvalues, axis=0).astype(np.float32)
            if padded_local_eigenvalues
            else np.zeros((n_components, 0), dtype=np.float32)
        )
        group_scores = _group_scores_numpy(
            group_features,
            means,
            inverse_covariances_np,
            log_determinants_np,
            weights,
            np.stack(padded_local_bases, axis=0) if padded_local_bases else np.zeros((n_components, group_features.shape[1], 0), dtype=np.float32),
            padded_local_eigenvalues_np,
            np.asarray(local_factor_residuals, dtype=np.float32),
            np.asarray(local_factor_ranks, dtype=np.int64),
            method.support_metric,
        )

        if group_label == -1:
            global_old_scores = group_scores.astype(np.float32)
            global_assignments = (local_assignments + component_offset).astype(np.int64)

        group_thresholds.append(float(np.quantile(group_scores, config.support_distance_quantile)))
        group_score_medians.append(float(np.median(group_scores)))

        component_means.extend(list(means))
        component_covariances.extend(list(covariances))
        component_inverse_covariances.extend(list(inverse_covariances_np))
        component_log_determinants.extend(log_determinants_np.tolist())
        component_weights.extend(weights.tolist())
        component_counts.extend(local_counts.tolist())
        component_group_ids.extend([group_index] * n_components)
        component_class_ids.extend([group_label] * n_components)
        pca_bases.extend(local_pca_bases)
        factor_eigenvalues.extend(padded_local_eigenvalues)
        factor_residual_variances.extend(local_factor_residuals)
        factor_ranks.extend(local_factor_ranks)
        assignment_histogram.extend(local_counts.tolist())
        component_offset += n_components

    max_basis_rank = max((basis.shape[1] for basis in pca_bases), default=0)
    padded_bases = []
    for basis in pca_bases:
        if basis.shape[1] == max_basis_rank:
            padded_bases.append(basis)
            continue
        pad_width = max_basis_rank - basis.shape[1]
        padded_bases.append(np.pad(basis, ((0, 0), (0, pad_width)), mode="constant"))
    max_factor_rank = max((values.shape[0] for values in factor_eigenvalues), default=0)
    padded_factor_eigenvalues = []
    for values in factor_eigenvalues:
        if values.shape[0] == max_factor_rank:
            padded_factor_eigenvalues.append(values.astype(np.float32))
        else:
            padded_factor_eigenvalues.append(np.pad(values.astype(np.float32), (0, max_factor_rank - values.shape[0]), mode="constant"))

    memory = ManifoldMemory(
        means=np.stack(component_means, axis=0).astype(np.float32),
        covariances=np.stack(component_covariances, axis=0).astype(np.float32),
        inverse_covariances=np.stack(component_inverse_covariances, axis=0).astype(np.float32),
        log_determinants=np.asarray(component_log_determinants, dtype=np.float32),
        mixture_weights=np.asarray(component_weights, dtype=np.float32),
        counts=np.asarray(component_counts, dtype=np.int64),
        pca_bases=np.stack(padded_bases, axis=0) if padded_bases else np.zeros((0, 0, 0), dtype=np.float32),
        factor_eigenvalues=np.stack(padded_factor_eigenvalues, axis=0) if padded_factor_eigenvalues else np.zeros((0, 0), dtype=np.float32),
        factor_residual_variances=np.asarray(factor_residual_variances, dtype=np.float32),
        factor_ranks=np.asarray(factor_ranks, dtype=np.int64),
        component_tangent_thresholds=np.asarray(component_tangent_thresholds, dtype=np.float32),
        component_tangent_medians=np.asarray(component_tangent_medians, dtype=np.float32),
        component_residual_thresholds=np.asarray(component_residual_thresholds, dtype=np.float32),
        component_residual_medians=np.asarray(component_residual_medians, dtype=np.float32),
        component_group_ids=np.asarray(component_group_ids, dtype=np.int64),
        component_class_ids=np.asarray(component_class_ids, dtype=np.int64),
        group_labels=np.asarray(group_labels, dtype=np.int64),
        group_thresholds=np.asarray(group_thresholds, dtype=np.float32),
        group_score_medians=np.asarray(group_score_medians, dtype=np.float32),
        global_threshold=float(np.quantile(global_old_scores, config.support_distance_quantile)),
        global_score_median=float(np.median(global_old_scores)),
        assignment_histogram=np.asarray(assignment_histogram, dtype=np.int64),
        support_metric=method.support_metric,
        manifold_builder=method.manifold_builder,
        conditioning=method.support_conditioning,
    )
    caches = {
        "anchor_features": anchor_features.astype(np.float32),
        "anchor_labels": anchor_labels.astype(np.int64),
        "assignments": global_assignments.astype(np.int64),
        "old_scores": global_old_scores.astype(np.float32),
    }
    return memory, caches


def build_manifold_memory(
    teacher: torch.nn.Module,
    anchor_dataset,
    method: MethodConfig,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[ManifoldMemory, dict[str, np.ndarray]]:
    anchor_features, anchor_labels = collect_latents(teacher, anchor_dataset, config.eval_batch_size, config, device)
    if anchor_features.shape[0] < 1:
        raise ValueError("Anchor dataset is empty; cannot build manifold memory.")
    return build_manifold_memory_from_latents(anchor_features, anchor_labels, method, config)


def build_manifold_memory_from_latents(
    anchor_features: np.ndarray,
    anchor_labels: np.ndarray,
    method: MethodConfig,
    config: ExperimentConfig,
) -> tuple[ManifoldMemory, dict[str, np.ndarray]]:
    if method.support_metric == "gmm_nll" and method.manifold_builder != "gmm":
        raise ValueError("GMM NLL support requires manifold_builder='gmm'.")
    return _build_memory_from_features(anchor_features, anchor_labels, method, config)


def build_multilayer_manifold_memories(
    teacher: torch.nn.Module,
    anchor_dataset,
    method: MethodConfig,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, ManifoldMemory], dict[str, dict[str, np.ndarray]], dict[str, LayerProjection]]:
    layer_names = tuple(method.multilayer_support_layers)
    if not layer_names:
        return {}, {}, {}

    layer_features, anchor_labels = collect_layer_latents(
        teacher,
        anchor_dataset,
        layer_names=layer_names,
        batch_size=config.eval_batch_size,
        config=config,
        device=device,
    )
    return build_multilayer_manifold_memories_from_latents(layer_features, anchor_labels, method, config)


def build_multilayer_manifold_memories_from_latents(
    layer_features: dict[str, np.ndarray],
    anchor_labels: np.ndarray,
    method: MethodConfig,
    config: ExperimentConfig,
) -> tuple[dict[str, ManifoldMemory], dict[str, dict[str, np.ndarray]], dict[str, LayerProjection]]:
    layer_names = tuple(method.multilayer_support_layers)
    if not layer_names:
        return {}, {}, {}

    layer_memories: dict[str, ManifoldMemory] = {}
    layer_caches: dict[str, dict[str, np.ndarray]] = {}
    layer_projections: dict[str, LayerProjection] = {}
    for layer_name in layer_names:
        projection = _fit_projection(layer_features[layer_name], projection_dim=config.multilayer_projection_dim)
        projected_features = project_features_numpy(layer_features[layer_name], projection)
        memory, caches = _build_memory_from_features(projected_features, anchor_labels, method, config)
        layer_memories[layer_name] = memory
        layer_caches[layer_name] = {
            **caches,
            "raw_anchor_features": layer_features[layer_name].astype(np.float32),
        }
        layer_projections[layer_name] = projection
    return layer_memories, layer_caches, layer_projections


def manifold_memory_to_torch(memory: ManifoldMemory, device: torch.device) -> TorchManifoldMemory:
    return TorchManifoldMemory(
        means=torch.from_numpy(memory.means).to(device),
        inverse_covariances=torch.from_numpy(memory.inverse_covariances).to(device),
        log_determinants=torch.from_numpy(memory.log_determinants).to(device),
        mixture_weights=torch.from_numpy(memory.mixture_weights).to(device),
        pca_bases=torch.from_numpy(memory.pca_bases).to(device),
        factor_eigenvalues=torch.from_numpy(memory.factor_eigenvalues).to(device),
        factor_residual_variances=torch.from_numpy(memory.factor_residual_variances).to(device),
        factor_ranks=torch.from_numpy(memory.factor_ranks).to(device),
        component_tangent_thresholds=torch.from_numpy(memory.component_tangent_thresholds).to(device),
        component_tangent_medians=torch.from_numpy(memory.component_tangent_medians).to(device),
        component_residual_thresholds=torch.from_numpy(memory.component_residual_thresholds).to(device),
        component_residual_medians=torch.from_numpy(memory.component_residual_medians).to(device),
        component_group_ids=torch.from_numpy(memory.component_group_ids).to(device),
        group_labels=torch.from_numpy(memory.group_labels).to(device),
        group_thresholds=torch.from_numpy(memory.group_thresholds).to(device),
        group_score_medians=torch.from_numpy(memory.group_score_medians).to(device),
        global_threshold=torch.tensor(memory.global_threshold, device=device),
        support_metric=memory.support_metric,
        conditioning=memory.conditioning,
    )


def layer_projection_to_torch(projection: LayerProjection, device: torch.device) -> TorchLayerProjection:
    return TorchLayerProjection(
        mean=torch.from_numpy(projection.mean).to(device),
        basis=torch.from_numpy(projection.basis).to(device),
    )
