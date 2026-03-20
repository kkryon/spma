from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .manifold_memory import TorchManifoldMemory


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl * (temperature**2)


def hidden_feature_l2(student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
    return (student_features - teacher_features).pow(2).mean()


def old_logit_suppression_loss(
    old_logits: torch.Tensor,
    full_logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    true_logits = full_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    old_mass = torch.logsumexp(old_logits, dim=1)
    return F.relu(old_mass - true_logits + margin).mean()


def normalized_pairwise_distance_matrix(features: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(features, features, p=2)
    off_diagonal = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
    scale = distances[off_diagonal].mean().clamp_min(1e-6)
    return distances / scale


def relational_geometry_loss(student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
    student_distances = normalized_pairwise_distance_matrix(student_features)
    teacher_distances = normalized_pairwise_distance_matrix(teacher_features)
    residual = (student_distances - teacher_distances).pow(2)
    off_diagonal = ~torch.eye(residual.size(0), dtype=torch.bool, device=residual.device)
    return residual[off_diagonal].mean()


def local_manifold_smoothing_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    knn_k: int,
    affinity_temperature: float,
) -> torch.Tensor:
    if student_features.size(0) <= 1:
        return torch.tensor(0.0, device=student_features.device)

    teacher_distances = normalized_pairwise_distance_matrix(teacher_features)
    student_distances = normalized_pairwise_distance_matrix(student_features)
    batch_size = teacher_distances.size(0)
    effective_k = min(knn_k, max(batch_size - 1, 1))
    if effective_k <= 0:
        return torch.tensor(0.0, device=student_features.device)

    masked_teacher_distances = teacher_distances + torch.eye(batch_size, device=teacher_distances.device) * 1e6
    neighbor_indices = masked_teacher_distances.topk(k=effective_k, largest=False).indices
    neighbor_mask = torch.zeros_like(teacher_distances, dtype=torch.bool)
    row_indices = torch.arange(batch_size, device=teacher_distances.device).unsqueeze(1).expand_as(neighbor_indices)
    neighbor_mask[row_indices, neighbor_indices] = True
    neighbor_mask = neighbor_mask | neighbor_mask.T

    weights = torch.exp(-teacher_distances / max(affinity_temperature, 1e-6))
    weights = weights * neighbor_mask
    residual = (student_distances - teacher_distances).pow(2)
    normalization = weights.sum().clamp_min(1e-6)
    return (weights * residual).sum() / normalization


def _component_distances(
    features: torch.Tensor,
    means: torch.Tensor,
    inverse_covariances: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    deltas = features[:, None, :] - means[None, :, :]
    if metric == "euclidean":
        return deltas.pow(2).mean(dim=-1)
    projected = torch.einsum("bkd,kde->bke", deltas, inverse_covariances)
    return (projected * deltas).sum(dim=-1) / max(features.size(-1), 1)


def component_score_matrix(
    features: torch.Tensor,
    memory: TorchManifoldMemory,
    metric: str,
) -> torch.Tensor:
    if metric == "gmm_nll":
        deltas = features[:, None, :] - memory.means[None, :, :]
        projected = torch.einsum("bkd,kde->bke", deltas, memory.inverse_covariances)
        mahalanobis = (projected * deltas).sum(dim=-1)
        dim = features.size(-1)
        log_prob = -0.5 * (dim * math.log(2.0 * math.pi) + memory.log_determinants.unsqueeze(0) + mahalanobis)
        weighted = log_prob + memory.mixture_weights.clamp_min(1e-12).log().unsqueeze(0)
        return -weighted / max(dim, 1)
    if metric == "factor_nll":
        return _factor_negative_log_likelihood(
            features,
            memory.means,
            memory.pca_bases,
            memory.factor_eigenvalues,
            memory.factor_residual_variances,
            memory.factor_ranks,
        )
    return _component_distances(features, memory.means, memory.inverse_covariances, metric)


def _gmm_negative_log_likelihood(
    features: torch.Tensor,
    means: torch.Tensor,
    inverse_covariances: torch.Tensor,
    log_determinants: torch.Tensor,
    mixture_weights: torch.Tensor,
) -> torch.Tensor:
    deltas = features[:, None, :] - means[None, :, :]
    projected = torch.einsum("bkd,kde->bke", deltas, inverse_covariances)
    mahalanobis = (projected * deltas).sum(dim=-1)
    dim = features.size(-1)
    log_prob = -0.5 * (dim * math.log(2.0 * math.pi) + log_determinants.unsqueeze(0) + mahalanobis)
    weighted = log_prob + mixture_weights.clamp_min(1e-12).log().unsqueeze(0)
    return -torch.logsumexp(weighted, dim=1) / max(dim, 1)


def _factor_negative_log_likelihood(
    features: torch.Tensor,
    means: torch.Tensor,
    bases: torch.Tensor,
    eigenvalues: torch.Tensor,
    residual_variances: torch.Tensor,
    ranks: torch.Tensor,
) -> torch.Tensor:
    deltas = features[:, None, :] - means[None, :, :]
    basis_count = bases.size(-1)
    if basis_count > 0:
        coefficients = torch.einsum("bkd,kdr->bkr", deltas, bases)
        reconstructed = torch.einsum("bkr,kdr->bkd", coefficients, bases)
        residuals = deltas - reconstructed
        active_mask = torch.arange(basis_count, device=features.device).unsqueeze(0) < ranks.unsqueeze(1)
        variances = eigenvalues.unsqueeze(0) + residual_variances.unsqueeze(0).unsqueeze(-1)
        factor_term = torch.where(
            active_mask.unsqueeze(0),
            coefficients.pow(2) / variances.clamp_min(1e-6),
            torch.zeros_like(coefficients),
        ).sum(dim=-1)
        logdet_factors = torch.where(
            active_mask,
            variances.squeeze(0).clamp_min(1e-6).log(),
            torch.zeros_like(variances.squeeze(0)),
        ).sum(dim=-1)
    else:
        residuals = deltas
        factor_term = torch.zeros((features.size(0), means.size(0)), device=features.device)
        logdet_factors = torch.zeros((means.size(0),), device=features.device)

    residual_term = residuals.pow(2).sum(dim=-1) / residual_variances.unsqueeze(0).clamp_min(1e-6)
    dim = features.size(-1)
    logdet_residual = (dim - ranks.to(features.dtype)) * residual_variances.clamp_min(1e-6).log()
    return 0.5 * (
        factor_term
        + residual_term
        + logdet_factors.unsqueeze(0)
        + logdet_residual.unsqueeze(0)
        + dim * math.log(2.0 * math.pi)
    ) / max(dim, 1)


def resolve_group_indices(labels: torch.Tensor | None, memory: TorchManifoldMemory) -> torch.Tensor:
    default_group = torch.zeros((), dtype=torch.long, device=memory.group_labels.device)
    if labels is None or memory.conditioning == "global":
        return torch.full((labels.size(0) if labels is not None else 0,), int(default_group.item()), dtype=torch.long, device=memory.group_labels.device) if labels is not None else torch.empty(0, dtype=torch.long, device=memory.group_labels.device)

    group_indices = torch.zeros_like(labels, device=memory.group_labels.device)
    group_indices.fill_(int(default_group.item()))
    for group_index, group_label in enumerate(memory.group_labels.tolist()):
        if group_label < 0:
            continue
        group_indices[labels.to(memory.group_labels.device) == group_label] = group_index
    return group_indices


def support_scores(
    features: torch.Tensor,
    memory: TorchManifoldMemory,
    metric: str,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels is None or memory.conditioning == "global":
        group_indices = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
    else:
        group_indices = resolve_group_indices(labels, memory)

    scores = torch.zeros(features.size(0), device=features.device)
    unique_groups = torch.unique(group_indices, sorted=True)
    for group_index in unique_groups.tolist():
        sample_mask = group_indices == group_index
        component_mask = memory.component_group_ids == group_index
        group_features = features[sample_mask]
        group_means = memory.means[component_mask]
        group_inverse_covariances = memory.inverse_covariances[component_mask]
        if metric == "gmm_nll":
            scores[sample_mask] = _gmm_negative_log_likelihood(
                group_features,
                group_means,
                group_inverse_covariances,
                memory.log_determinants[component_mask],
                memory.mixture_weights[component_mask],
            )
        elif metric == "factor_nll":
            scores[sample_mask] = _factor_negative_log_likelihood(
                group_features,
                group_means,
                memory.pca_bases[component_mask],
                memory.factor_eigenvalues[component_mask],
                memory.factor_residual_variances[component_mask],
                memory.factor_ranks[component_mask],
            ).min(dim=1).values
        else:
            scores[sample_mask] = _component_distances(group_features, group_means, group_inverse_covariances, metric).min(dim=1).values
    return scores, group_indices


def nearest_cluster_support_distance(
    features: torch.Tensor,
    memory: TorchManifoldMemory,
    metric: str,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores, group_indices = support_scores(features, memory, metric=metric, labels=labels)
    return scores, group_indices


def soft_chart_assignment_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    memory: TorchManifoldMemory,
    metric: str,
    temperature: float,
) -> torch.Tensor:
    teacher_scores = component_score_matrix(teacher_features, memory, metric=metric)
    student_scores = component_score_matrix(student_features, memory, metric=metric)
    teacher_probs = F.softmax(-teacher_scores / temperature, dim=-1)
    student_log_probs = F.log_softmax(-student_scores / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def local_chart_continuation_loss(
    features: torch.Tensor,
    memory: TorchManifoldMemory,
    metric: str,
    tangent_expansion_scale: float,
    residual_expansion_scale: float,
    tangent_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    component_scores = component_score_matrix(features, memory, metric=metric)
    nearest_components = component_scores.argmin(dim=1)
    means = memory.means[nearest_components]
    bases = memory.pca_bases[nearest_components]
    eigenvalues = memory.factor_eigenvalues[nearest_components]
    residual_variances = memory.factor_residual_variances[nearest_components].clamp_min(1e-6)
    ranks = memory.factor_ranks[nearest_components]

    centered = features - means
    max_rank = bases.size(-1)
    if max_rank > 0:
        coefficients = torch.einsum("bd,bdr->br", centered, bases)
        active_mask = torch.arange(max_rank, device=features.device).unsqueeze(0) < ranks.unsqueeze(1)
        variances = (eigenvalues + residual_variances.unsqueeze(1)).clamp_min(1e-6)
        tangent_scores = torch.where(active_mask, coefficients.pow(2) / variances, torch.zeros_like(coefficients)).sum(dim=-1)
        tangent_scores = tangent_scores / ranks.clamp_min(1).to(features.dtype)
        reconstructed = torch.einsum("br,bdr->bd", coefficients, bases)
        residuals = centered - reconstructed
    else:
        tangent_scores = torch.zeros(features.size(0), device=features.device)
        residuals = centered

    residual_scores = residuals.pow(2).mean(dim=-1) / residual_variances

    tangent_thresholds = memory.component_tangent_thresholds[nearest_components]
    tangent_medians = memory.component_tangent_medians[nearest_components]
    residual_thresholds = memory.component_residual_thresholds[nearest_components]
    residual_medians = memory.component_residual_medians[nearest_components]

    tangent_shell = tangent_expansion_scale * (tangent_thresholds - tangent_medians).clamp_min(1e-3)
    residual_shell = residual_expansion_scale * (residual_thresholds - residual_medians).clamp_min(1e-3)
    tangent_penalty = F.relu(tangent_scores - tangent_thresholds - tangent_shell)
    residual_penalty = F.relu(residual_scores - residual_thresholds - residual_shell)
    penalties = residual_penalty + tangent_weight * tangent_penalty
    return penalties.mean(), {
        "tangent_penalty": tangent_penalty.mean(),
        "residual_penalty": residual_penalty.mean(),
        "tangent_score": tangent_scores.mean(),
        "residual_score": residual_scores.mean(),
    }


def support_loss(
    features: torch.Tensor,
    memory: TorchManifoldMemory,
    metric: str,
    labels: torch.Tensor | None = None,
    loss_mode: str = "strict",
    expansion_scale: float = 1.25,
    novelty_gate_center: float = 0.5,
    novelty_temperature: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores, group_indices = support_scores(features, memory, metric=metric, labels=labels)
    thresholds = memory.group_thresholds[group_indices]
    if loss_mode == "strict":
        penalties = F.relu(scores - thresholds)
        return penalties.mean(), scores, group_indices

    if loss_mode != "bounded_expansion":
        raise ValueError(f"Unsupported support loss mode: {loss_mode}")

    medians = memory.group_score_medians[group_indices]
    shell_width = expansion_scale * (thresholds - medians).clamp_min(1e-3)
    strict_penalties = F.relu(scores - thresholds)
    expansion_penalties = F.relu(scores - thresholds - shell_width)
    normalized_excess = (scores - thresholds) / shell_width.clamp_min(1e-6)
    novelty_gate = torch.sigmoid((normalized_excess - novelty_gate_center) / max(novelty_temperature, 1e-6))
    penalties = (1.0 - novelty_gate) * strict_penalties + novelty_gate * expansion_penalties
    return penalties.mean(), scores, group_indices


def subspace_projection_error(
    features: torch.Tensor,
    memory: TorchManifoldMemory,
    group_indices: torch.Tensor,
) -> torch.Tensor:
    basis = memory.pca_bases[group_indices]
    means = memory.means[group_indices]
    centered = features - means
    if basis.size(-1) == 0:
        return centered.pow(2).mean(dim=-1)
    projected = torch.einsum("bd,bdr->br", centered, basis)
    reconstructed = torch.einsum("br,bdr->bd", projected, basis)
    return (centered - reconstructed).pow(2).mean(dim=-1)


def parameter_drift_penalty(
    model: torch.nn.Module,
    reference_state: dict[str, torch.Tensor],
    exempt_prefixes: tuple[str, ...] = (),
) -> torch.Tensor:
    penalties: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in exempt_prefixes):
            continue
        if name not in reference_state:
            continue
        reference = reference_state[name].to(parameter.device)
        penalties.append((parameter - reference).pow(2).mean())
    return torch.stack(penalties).mean() if penalties else torch.tensor(0.0, device=next(model.parameters()).device)
