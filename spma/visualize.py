from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import ExperimentConfig
from .manifold_memory import ManifoldMemory


def _class_color(class_id: int) -> Any:
    cmap = plt.get_cmap("tab20")
    return cmap(class_id % 20)


def _draw_covariance_ellipse(axis: plt.Axes, mean: np.ndarray, covariance: np.ndarray, color: Any) -> None:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.0 * np.sqrt(np.maximum(eigenvalues[:2], 1e-6))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color, facecolor="none", linewidth=1.0, alpha=0.8)
    axis.add_patch(ellipse)


def _balanced_subsample(
    features: np.ndarray,
    labels: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if features.shape[0] <= max_points:
        return features, labels

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    per_label = max(1, max_points // max(1, unique_labels.shape[0]))
    selected_indices: list[np.ndarray] = []
    for label in unique_labels:
        label_indices = np.flatnonzero(labels == label)
        take = min(per_label, label_indices.shape[0])
        selected_indices.append(rng.choice(label_indices, size=take, replace=False))

    merged = np.concatenate(selected_indices)
    if merged.shape[0] > max_points:
        merged = rng.choice(merged, size=max_points, replace=False)
    elif merged.shape[0] < max_points:
        remaining = np.setdiff1d(np.arange(features.shape[0]), merged, assume_unique=False)
        if remaining.shape[0] > 0:
            extra = rng.choice(remaining, size=min(max_points - merged.shape[0], remaining.shape[0]), replace=False)
            merged = np.concatenate([merged, extra])

    merged.sort()
    return features[merged], labels[merged]


def plot_latent_tsne_triptych(
    teacher_caches: dict[str, Any],
    plain_caches: dict[str, Any],
    spma_caches: dict[str, Any],
    memory: ManifoldMemory,
    config: ExperimentConfig,
    save_path: str | Path,
) -> None:
    max_points_per_group = 1500
    teacher_old_features, teacher_old_labels = _balanced_subsample(
        teacher_caches["viz_teacher_old"].features,
        teacher_caches["viz_teacher_old"].labels,
        max_points=max_points_per_group,
        seed=config.seed + 301,
    )
    teacher_new_features, teacher_new_labels = _balanced_subsample(
        teacher_caches["viz_teacher_new"].features,
        teacher_caches["viz_teacher_new"].labels,
        max_points=max_points_per_group,
        seed=config.seed + 302,
    )
    plain_old_features, plain_old_labels = _balanced_subsample(
        plain_caches["viz_student_old"].features,
        plain_caches["viz_student_old"].labels,
        max_points=max_points_per_group,
        seed=config.seed + 303,
    )
    plain_new_features, plain_new_labels = _balanced_subsample(
        plain_caches["viz_student_new"].features,
        plain_caches["viz_student_new"].labels,
        max_points=max_points_per_group,
        seed=config.seed + 304,
    )
    spma_old_features, spma_old_labels = _balanced_subsample(
        spma_caches["viz_student_old"].features,
        spma_caches["viz_student_old"].labels,
        max_points=max_points_per_group,
        seed=config.seed + 305,
    )
    spma_new_features, spma_new_labels = _balanced_subsample(
        spma_caches["viz_student_new"].features,
        spma_caches["viz_student_new"].labels,
        max_points=max_points_per_group,
        seed=config.seed + 306,
    )

    stacked = np.concatenate(
        [
            teacher_old_features,
            teacher_new_features,
            plain_old_features,
            plain_new_features,
            spma_old_features,
            spma_new_features,
            memory.means,
        ],
        axis=0,
    )
    perplexity = min(config.tsne_perplexity, max(5.0, stacked.shape[0] / 10.0))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=config.seed,
    ).fit_transform(stacked)

    counts = [
        teacher_old_features.shape[0],
        teacher_new_features.shape[0],
        plain_old_features.shape[0],
        plain_new_features.shape[0],
        spma_old_features.shape[0],
        spma_new_features.shape[0],
        memory.means.shape[0],
    ]
    offsets = np.cumsum([0] + counts)
    sections = [embedding[offsets[index] : offsets[index + 1]] for index in range(len(counts))]
    teacher_old_emb, teacher_new_emb, plain_old_emb, plain_new_emb, spma_old_emb, spma_new_emb, center_emb = sections

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    panels = [
        ("Teacher Before FT", axes[0], teacher_old_emb, teacher_old_labels, teacher_new_emb, teacher_new_labels),
        ("Plain FT", axes[1], plain_old_emb, plain_old_labels, plain_new_emb, plain_new_labels),
        ("SPMA", axes[2], spma_old_emb, spma_old_labels, spma_new_emb, spma_new_labels),
    ]
    for title, axis, old_emb, old_labels, new_emb, new_labels in panels:
        for class_id in np.unique(old_labels):
            mask = old_labels == class_id
            axis.scatter(old_emb[mask, 0], old_emb[mask, 1], s=14, alpha=0.75, color=_class_color(int(class_id)), label=f"old {class_id}")
        for class_id in np.unique(new_labels):
            mask = new_labels == class_id
            axis.scatter(new_emb[mask, 0], new_emb[mask, 1], s=20, alpha=0.85, marker="x", color=_class_color(int(class_id)), label=f"new {class_id}")
        axis.scatter(center_emb[:, 0], center_emb[:, 1], s=85, marker="*", color="black", label="cluster center")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    handles, labels = axes[-1].get_legend_handles_labels()
    legend_map = dict(zip(labels, handles, strict=False))
    axes[-1].legend(legend_map.values(), legend_map.keys(), fontsize=8, ncol=2, loc="best")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_support_pca(
    teacher_caches: dict[str, Any],
    plain_caches: dict[str, Any],
    spma_caches: dict[str, Any],
    memory: ManifoldMemory,
    save_path: str | Path,
) -> None:
    max_anchor_points = 2000
    max_new_points = 2000
    anchor_features, anchor_labels = _balanced_subsample(
        teacher_caches["anchor_teacher"].features,
        teacher_caches["anchor_teacher"].labels,
        max_points=max_anchor_points,
        seed=401,
    )
    plain_new_features, plain_new_labels = _balanced_subsample(
        plain_caches["viz_student_new"].features,
        plain_caches["viz_student_new"].labels,
        max_points=max_new_points,
        seed=402,
    )
    spma_new_features, spma_new_labels = _balanced_subsample(
        spma_caches["viz_student_new"].features,
        spma_caches["viz_student_new"].labels,
        max_points=max_new_points,
        seed=403,
    )

    stacked = np.concatenate(
        [
            anchor_features,
            plain_new_features,
            spma_new_features,
            memory.means,
        ],
        axis=0,
    )
    pca = PCA(n_components=2, random_state=0)
    embedding = pca.fit_transform(stacked)
    counts = [
        anchor_features.shape[0],
        plain_new_features.shape[0],
        spma_new_features.shape[0],
        memory.means.shape[0],
    ]
    offsets = np.cumsum([0] + counts)
    anchor_emb = embedding[offsets[0] : offsets[1]]
    plain_new_emb = embedding[offsets[1] : offsets[2]]
    spma_new_emb = embedding[offsets[2] : offsets[3]]
    center_emb = embedding[offsets[3] : offsets[4]]

    projected_covariances = []
    for covariance in memory.covariances:
        projected_covariances.append(pca.components_ @ covariance @ pca.components_.T)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, title, new_emb, labels in [
        (axes[0], "Plain FT vs Old Support", plain_new_emb, plain_new_labels),
        (axes[1], "SPMA vs Old Support", spma_new_emb, spma_new_labels),
    ]:
        axis.scatter(anchor_emb[:, 0], anchor_emb[:, 1], s=14, alpha=0.25, color="#4c78a8", label="old anchors")
        for class_id in np.unique(labels):
            mask = labels == class_id
            axis.scatter(new_emb[mask, 0], new_emb[mask, 1], s=20, alpha=0.8, marker="x", color=_class_color(int(class_id)), label=f"new {class_id}")
        for index, projected_covariance in enumerate(projected_covariances):
            _draw_covariance_ellipse(axis, center_emb[index], projected_covariance, color="black")
        axis.scatter(center_emb[:, 0], center_emb[:, 1], s=80, marker="*", color="black", label="cluster center")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    handles, labels = axes[-1].get_legend_handles_labels()
    legend_map = dict(zip(labels, handles, strict=False))
    axes[-1].legend(legend_map.values(), legend_map.keys(), fontsize=8, ncol=2, loc="best")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_support_distance_histograms(
    plain_metrics: dict[str, Any],
    spma_metrics: dict[str, Any],
    save_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, title, metrics in [
        (axes[0], "Plain FT Support Distances", plain_metrics),
        (axes[1], "SPMA Support Distances", spma_metrics),
    ]:
        axis.hist(metrics["support_before_details"]["distances"], bins=25, alpha=0.55, label="before FT", color="#9ecae1")
        axis.hist(metrics["support_after_details"]["distances"], bins=25, alpha=0.55, label="after FT", color="#f28e2b")
        axis.set_title(title)
        axis.set_xlabel("Nearest-cluster support distance")
        axis.set_ylabel("Count")
        axis.legend(loc="best")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_metric_bars(method_rows: list[dict[str, Any]], save_path: str | Path) -> None:
    metrics = [
        "old_accuracy_after_ft",
        "new_accuracy_after_ft",
        "cka",
        "support_distance_after_ft",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    method_names = [row["method_name"] for row in method_rows]
    for axis, metric in zip(axes, metrics, strict=True):
        axis.bar(method_names, [row[metric] for row in method_rows], color="#4c78a8")
        axis.set_title(metric.replace("_", " "))
        axis.tick_params(axis="x", rotation=35)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
