from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class MethodConfig:
    name: str
    description: str
    lambda_kd: float = 0.0
    lambda_anchor_ce: float = 0.0
    lambda_geo: float = 0.0
    lambda_smooth: float = 0.0
    lambda_support: float = 0.0
    lambda_reg: float = 0.0
    lambda_hidden: float = 0.0
    lambda_chart: float = 0.0
    lambda_continue: float = 0.0
    lambda_old_suppression: float = 0.0
    temperature: float = 2.0
    chart_temperature: float = 0.5
    support_metric: str = "mahalanobis"
    support_loss_mode: str = "strict"
    support_conditioning: str = "global"
    manifold_builder: str = "kmeans"
    trainable_mode: str = "full"
    classifier_mode: str = "auto"
    anchor_sampling_mode: str = "full_anchor"
    step_mix_mode: str = "joint"
    ce_step_ratio: float = 1.0
    multilayer_support_layers: tuple[str, ...] = ()
    multilayer_support_weights: tuple[float, ...] = ()
    new_task_loss_mode: str = "full_ce"
    old_suppression_margin: float = 0.0
    continuation_tangent_scale: float = 1.5
    continuation_residual_scale: float = 0.5
    continuation_tangent_weight: float = 0.25
    retention_schedule: str = "constant"
    retention_final_scale: float = 1.0
    anchor_ce_final_scale: float = 1.0
    calibration_mode: str = "none"
    calibration_old_budget: int | None = None
    calibration_new_budget: int | None = None
    calibration_epochs: int = 0
    calibration_learning_rate: float = 5e-2
    posthoc_refine_mode: str = "none"
    posthoc_refine_old_budget: int | None = None
    posthoc_refine_new_budget: int | None = None
    posthoc_refine_epochs: int = 0
    posthoc_refine_learning_rate: float = 1e-3
    anchor_buffer_per_class_override: int | None = None
    anchor_total_budget_override: int | None = None
    num_clusters_override: int | None = None
    lora_rank_override: int | None = None
    lora_alpha_override: float | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    benchmark_name: str = "split_mnist"
    seed: int = 7
    device: str = "auto"
    deterministic: bool = True
    fast_gpu_mode: bool = False
    enable_amp: bool = False
    allow_tf32: bool = False
    cudnn_benchmark: bool = False
    channels_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 4
    cache_teacher_features: bool = True
    old_classes: tuple[int, ...] = (0, 1, 2, 3, 4)
    new_classes: tuple[int, ...] = (5, 6, 7, 8, 9)
    num_classes: int = 10
    batch_size: int = 256
    anchor_batch_size: int = 96
    eval_batch_size: int = 512
    num_workers: int = 0
    base_epochs: int = 6
    finetune_epochs: int = 5
    base_learning_rate: float = 1e-3
    finetune_learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    latent_dim: int = 64
    backbone_name: str = "auto"
    anchor_buffer_per_class: int = 64
    anchor_eval_per_class: int = 200
    visualization_per_class: int = 80
    num_clusters: int = 8
    cluster_pca_rank: int = 8
    covariance_eps: float = 1e-3
    gmm_reg_covar: float = 1e-4
    hdbscan_min_cluster_size: int = 12
    hdbscan_min_samples: int | None = None
    hdbscan_cluster_selection_epsilon: float = 0.0
    hdbscan_allow_single_cluster: bool = True
    support_distance_quantile: float = 0.95
    support_confidence: float = 0.95
    support_expansion_scale: float = 1.25
    support_novelty_gate_center: float = 0.5
    support_novelty_temperature: float = 0.2
    smooth_knn_k: int = 6
    smooth_affinity_temperature: float = 1.0
    multilayer_projection_dim: int = 64
    tsne_perplexity: float = 30.0
    knn_k: int = 10
    lora_rank: int = 8
    lora_alpha: float = 8.0
    dual_head_warmup_epochs: int = 2
    dual_head_feature_lr_scale: float = 0.2
    dual_head_old_head_lr_scale: float = 0.2
    dual_head_new_head_lr_scale: float = 1.0
    dual_head_new_head_bias_init: float = -8.0
    compatible_shift_rotation_deg: float = 12.0
    compatible_shift_translate_x: int = 2
    compatible_shift_translate_y: int = -1
    compatible_shift_scale: float = 0.95
    compatible_shift_noise_std: float = 0.03
    compatible_shift_blur_kernel: int = 3
    stress_shift_rotation_deg: float = 35.0
    stress_shift_translate_x: int = 4
    stress_shift_translate_y: int = -3
    stress_shift_scale: float = 0.9
    stress_shift_noise_std: float = 0.06
    stress_shift_blur_kernel: int = 5
    enable_plots: bool = True
    enable_ablations: bool = True


def build_default_config(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs/spma/latest",
    checkpoint_dir: str | Path = "checkpoints/spma/latest",
) -> ExperimentConfig:
    return ExperimentConfig(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        checkpoint_dir=Path(checkpoint_dir),
    )


def build_baseline_methods(config: ExperimentConfig) -> list[MethodConfig]:
    del config
    base_methods = [
        MethodConfig(
            name="plain_ft",
            description="Plain fine-tuning on Task 2 with only new-task cross-entropy.",
        ),
        MethodConfig(
            name="kd_only",
            description="Fine-tuning with old-task logit distillation on the anchor buffer.",
            lambda_kd=1.0,
        ),
        MethodConfig(
            name="ce90_kd10_stepmix",
            description="Fine-tuning initialized from the teacher with 90% CE(new) steps and 10% KD(anchor, teacher) steps.",
            lambda_kd=1.0,
            step_mix_mode="ce_kd_stepmix",
            ce_step_ratio=0.9,
        ),
        MethodConfig(
            name="anchor_ce_only",
            description="Fine-tuning with old-anchor supervised cross-entropy on a sparse replay set.",
            lambda_anchor_ce=1.0,
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=256,
        ),
        MethodConfig(
            name="er_replay_128",
            description="Experience replay baseline with 128 sparse old anchors and supervised CE on replayed samples.",
            lambda_anchor_ce=1.0,
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=128,
        ),
        MethodConfig(
            name="er_replay_256",
            description="Experience replay baseline with 256 sparse old anchors and supervised CE on replayed samples.",
            lambda_anchor_ce=1.0,
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=256,
        ),
        MethodConfig(
            name="er_replay_512",
            description="Experience replay baseline with 512 sparse old anchors and supervised CE on replayed samples.",
            lambda_anchor_ce=1.0,
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
        ),
        MethodConfig(
            name="spma_simple_512",
            description="Simple manifold baseline: CE(new head) + CE(old sparse anchors) + bounded support loss.",
            lambda_anchor_ce=1.0,
            lambda_support=0.02,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
        ),
        MethodConfig(
            name="spma_simple_factor_512",
            description="Simple manifold baseline with local factor support: CE(new head) + CE(old sparse anchors) + factor support loss.",
            lambda_anchor_ce=1.0,
            lambda_support=0.02,
            support_metric="factor_nll",
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
        ),
        MethodConfig(
            name="spma_simple_decay_512",
            description="Minimal viable manifold method: CE(new head) + CE(old sparse anchors) + support, with late decay of old-pressure terms.",
            lambda_anchor_ce=1.0,
            lambda_support=0.02,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
            retention_schedule="linear_decay",
            retention_final_scale=0.05,
            anchor_ce_final_scale=0.2,
        ),
        MethodConfig(
            name="spma_simple_factor_decay_512",
            description="Minimal viable manifold method with factor support: CE(new head) + CE(old sparse anchors) + factor support, with late decay of old-pressure terms.",
            lambda_anchor_ce=1.0,
            lambda_support=0.02,
            support_metric="factor_nll",
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
            retention_schedule="linear_decay",
            retention_final_scale=0.05,
            anchor_ce_final_scale=0.2,
        ),
        MethodConfig(
            name="hidden_l2",
            description="Fine-tuning with coordinate-level hidden-state matching on anchors.",
            lambda_hidden=1.0,
        ),
        MethodConfig(
            name="geometry_only",
            description="Fine-tuning with relational anchor geometry preservation.",
            lambda_geo=0.5,
        ),
        MethodConfig(
            name="support_only",
            description="Fine-tuning with old-support assimilation for new-task embeddings.",
            lambda_support=0.1,
        ),
        MethodConfig(
            name="support_only_bounded",
            description="Fine-tuning with novelty-gated bounded support expansion for new-task embeddings.",
            lambda_support=0.05,
            support_loss_mode="bounded_expansion",
        ),
        MethodConfig(
            name="spma_full",
            description="Support-Preserving Manifold Assimilation: CE + KD + geometry + support + parameter regularization.",
            lambda_kd=1.0,
            lambda_geo=0.5,
            lambda_support=0.1,
            lambda_reg=0.005,
        ),
        MethodConfig(
            name="spma_novel",
            description="SPMA for novel classes: dual-head finetuning with novelty-gated bounded support expansion.",
            lambda_kd=1.0,
            lambda_geo=0.5,
            lambda_support=0.05,
            lambda_reg=0.005,
            support_loss_mode="bounded_expansion",
        ),
        MethodConfig(
            name="spma_sparse_smooth",
            description="SPMA with sparse cluster-guided anchors, old-anchor CE, and local manifold smoothing.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.25,
            lambda_smooth=0.5,
            lambda_support=0.05,
            lambda_reg=0.005,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=256,
        ),
        MethodConfig(
            name="spma_sparse_smooth_tuned_256",
            description="Tuned sparse-anchor SPMA preset with a 256-sample replay budget.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.25,
            lambda_smooth=0.1,
            lambda_support=0.05,
            lambda_reg=0.005,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=256,
        ),
        MethodConfig(
            name="spma_sparse_smooth_tuned_512",
            description="Tuned sparse-anchor SPMA preset with a 512-sample replay budget.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.25,
            lambda_smooth=0.1,
            lambda_support=0.02,
            lambda_reg=0.005,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
        ),
        MethodConfig(
            name="spma_sparse_smooth_boosted_512",
            description="Sparse-anchor SPMA with decayed retention losses and post-finetune dual-head calibration for higher novel-class accuracy.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.25,
            lambda_smooth=0.1,
            lambda_support=0.02,
            lambda_reg=0.005,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            retention_schedule="linear_decay",
            retention_final_scale=0.15,
            anchor_ce_final_scale=0.35,
            calibration_mode="balanced_affine",
            calibration_old_budget=256,
            calibration_new_budget=512,
            calibration_epochs=40,
            calibration_learning_rate=5e-2,
        ),
        MethodConfig(
            name="spma_sparse_smooth_decoupled_512",
            description="Sparse-anchor SPMA with decoupled new-head learning, old-logit suppression, balanced head refinement, and calibration.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.15,
            lambda_smooth=0.05,
            lambda_support=0.01,
            lambda_reg=0.0025,
            lambda_old_suppression=0.5,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
            old_suppression_margin=0.5,
            retention_schedule="linear_decay",
            retention_final_scale=0.05,
            anchor_ce_final_scale=0.2,
            posthoc_refine_mode="balanced_heads",
            posthoc_refine_old_budget=512,
            posthoc_refine_new_budget=1024,
            posthoc_refine_epochs=4,
            posthoc_refine_learning_rate=2e-3,
            calibration_mode="balanced_affine",
            calibration_old_budget=256,
            calibration_new_budget=512,
            calibration_epochs=40,
            calibration_learning_rate=5e-2,
        ),
        MethodConfig(
            name="spma_sparse_smooth_decoupled_512_hi_new",
            description="More aggressive novel-class SPMA variant with stronger decoupled new-head learning and weaker late retention.",
            lambda_kd=0.75,
            lambda_anchor_ce=0.75,
            lambda_geo=0.1,
            lambda_smooth=0.05,
            lambda_support=0.005,
            lambda_reg=0.001,
            lambda_old_suppression=0.75,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
            old_suppression_margin=0.75,
            retention_schedule="linear_decay",
            retention_final_scale=0.0,
            anchor_ce_final_scale=0.1,
            posthoc_refine_mode="balanced_heads",
            posthoc_refine_old_budget=512,
            posthoc_refine_new_budget=1536,
            posthoc_refine_epochs=5,
            posthoc_refine_learning_rate=3e-3,
            calibration_mode="balanced_affine",
            calibration_old_budget=256,
            calibration_new_budget=768,
            calibration_epochs=50,
            calibration_learning_rate=5e-2,
        ),
        MethodConfig(
            name="spma_sparse_smooth_decoupled_raw_512",
            description="Decoupled novel-class SPMA without posthoc correction, keeping the stronger end-of-training new-task fit.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.15,
            lambda_smooth=0.05,
            lambda_support=0.01,
            lambda_reg=0.0025,
            lambda_old_suppression=0.5,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
            old_suppression_margin=0.5,
            retention_schedule="linear_decay",
            retention_final_scale=0.05,
            anchor_ce_final_scale=0.2,
        ),
        MethodConfig(
            name="spma_sparse_smooth_decoupled_raw_plus_512",
            description="Slightly more new-task-biased raw decoupled SPMA without posthoc correction.",
            lambda_kd=0.85,
            lambda_anchor_ce=0.85,
            lambda_geo=0.1,
            lambda_smooth=0.05,
            lambda_support=0.005,
            lambda_reg=0.001,
            lambda_old_suppression=0.6,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
            old_suppression_margin=0.6,
            retention_schedule="linear_decay",
            retention_final_scale=0.02,
            anchor_ce_final_scale=0.15,
        ),
        MethodConfig(
            name="spma_sparse_smooth_decoupled_raw_calnew_512",
            description="Raw decoupled SPMA with a lightweight new-biased affine head calibration pass.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.15,
            lambda_smooth=0.05,
            lambda_support=0.01,
            lambda_reg=0.0025,
            lambda_old_suppression=0.5,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
            old_suppression_margin=0.5,
            retention_schedule="linear_decay",
            retention_final_scale=0.05,
            anchor_ce_final_scale=0.2,
            calibration_mode="balanced_affine",
            calibration_old_budget=64,
            calibration_new_budget=1536,
            calibration_epochs=50,
            calibration_learning_rate=5e-2,
        ),
        MethodConfig(
            name="spma_sparse_smooth_multilayer_512",
            description="Sparse-anchor SPMA with support memories on selected early, middle, and penultimate layers.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.2,
            lambda_smooth=0.1,
            lambda_support=0.02,
            lambda_reg=0.005,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            retention_schedule="linear_decay",
            retention_final_scale=0.2,
            anchor_ce_final_scale=0.4,
            calibration_mode="balanced_affine",
            calibration_old_budget=256,
            calibration_new_budget=512,
            calibration_epochs=40,
            calibration_learning_rate=5e-2,
            multilayer_support_layers=("layer1", "layer3", "penultimate"),
            multilayer_support_weights=(0.2, 0.3, 0.5),
        ),
        MethodConfig(
            name="spma_sparse_smooth_factor_512",
            description="Sparse-anchor SPMA with local factor-analyzer style support instead of plain Mahalanobis support.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.25,
            lambda_smooth=0.1,
            lambda_support=0.02,
            lambda_reg=0.005,
            support_metric="factor_nll",
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            retention_schedule="linear_decay",
            retention_final_scale=0.15,
            anchor_ce_final_scale=0.35,
            calibration_mode="balanced_affine",
            calibration_old_budget=256,
            calibration_new_budget=512,
            calibration_epochs=40,
            calibration_learning_rate=5e-2,
        ),
        MethodConfig(
            name="spma_sparse_smooth_decoupled_multilayer_512",
            description="Decoupled novel-class SPMA with weighted support memories on selected early, middle, and penultimate layers.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.15,
            lambda_smooth=0.05,
            lambda_support=0.02,
            lambda_reg=0.0025,
            lambda_old_suppression=0.5,
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
            old_suppression_margin=0.5,
            retention_schedule="linear_decay",
            retention_final_scale=0.05,
            anchor_ce_final_scale=0.2,
            calibration_mode="none",
            multilayer_support_layers=("layer1", "layer3", "penultimate"),
            multilayer_support_weights=(0.2, 0.3, 0.5),
        ),
        MethodConfig(
            name="spma_sparse_smooth_decoupled_factor_512",
            description="Decoupled novel-class SPMA with local factor-analyzer style support.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.15,
            lambda_smooth=0.05,
            lambda_support=0.02,
            lambda_reg=0.0025,
            lambda_old_suppression=0.5,
            support_metric="factor_nll",
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
            old_suppression_margin=0.5,
            retention_schedule="linear_decay",
            retention_final_scale=0.05,
            anchor_ce_final_scale=0.2,
            calibration_mode="none",
        ),
        MethodConfig(
            name="spma_manifold_continuation_512",
            description="Constrained manifold continuation with old chart-assignment preservation and local tangent-only continuation for new features.",
            lambda_kd=0.75,
            lambda_anchor_ce=1.0,
            lambda_geo=0.05,
            lambda_smooth=0.05,
            lambda_support=0.01,
            lambda_reg=0.0015,
            lambda_chart=0.5,
            lambda_continue=1.0,
            lambda_old_suppression=0.35,
            temperature=2.0,
            chart_temperature=0.5,
            support_metric="factor_nll",
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
            old_suppression_margin=0.4,
            continuation_tangent_scale=1.75,
            continuation_residual_scale=0.35,
            continuation_tangent_weight=0.2,
            retention_schedule="linear_decay",
            retention_final_scale=0.1,
            anchor_ce_final_scale=0.25,
            calibration_mode="balanced_affine",
            calibration_old_budget=128,
            calibration_new_budget=768,
            calibration_epochs=25,
            calibration_learning_rate=2e-2,
        ),
        MethodConfig(
            name="spma_manifold_continuation_hi_new_512",
            description="More new-task-biased constrained manifold continuation with a looser tangent shell and weaker old retention.",
            lambda_kd=0.5,
            lambda_anchor_ce=0.85,
            lambda_geo=0.0,
            lambda_smooth=0.05,
            lambda_support=0.005,
            lambda_reg=0.001,
            lambda_chart=0.35,
            lambda_continue=0.85,
            lambda_old_suppression=0.45,
            temperature=2.0,
            chart_temperature=0.5,
            support_metric="factor_nll",
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="new_head_ce",
            old_suppression_margin=0.5,
            continuation_tangent_scale=2.0,
            continuation_residual_scale=0.25,
            continuation_tangent_weight=0.15,
            retention_schedule="linear_decay",
            retention_final_scale=0.03,
            anchor_ce_final_scale=0.15,
            calibration_mode="balanced_affine",
            calibration_old_budget=64,
            calibration_new_budget=1024,
            calibration_epochs=25,
            calibration_learning_rate=2e-2,
        ),
        MethodConfig(
            name="spma_manifold_continuation_balanced_512",
            description="Balanced constrained manifold continuation using full-task CE, chart preservation, and local tangent continuation without explicit old-logit suppression.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.05,
            lambda_smooth=0.05,
            lambda_support=0.005,
            lambda_reg=0.002,
            lambda_chart=0.35,
            lambda_continue=0.75,
            temperature=2.0,
            chart_temperature=0.75,
            support_metric="factor_nll",
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="full_ce",
            continuation_tangent_scale=1.75,
            continuation_residual_scale=0.25,
            continuation_tangent_weight=0.15,
            retention_schedule="linear_decay",
            retention_final_scale=0.2,
            anchor_ce_final_scale=0.4,
            calibration_mode="balanced_affine",
            calibration_old_budget=128,
            calibration_new_budget=768,
            calibration_epochs=25,
            calibration_learning_rate=2e-2,
        ),
        MethodConfig(
            name="spma_manifold_continuation_balanced_hi_new_512",
            description="Higher-new-accuracy balanced continuation with weaker retention weights but the same shared-manifold continuation losses.",
            lambda_kd=0.75,
            lambda_anchor_ce=0.9,
            lambda_geo=0.0,
            lambda_smooth=0.05,
            lambda_support=0.003,
            lambda_reg=0.001,
            lambda_chart=0.3,
            lambda_continue=0.65,
            temperature=2.0,
            chart_temperature=0.75,
            support_metric="factor_nll",
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="full_ce",
            continuation_tangent_scale=2.0,
            continuation_residual_scale=0.2,
            continuation_tangent_weight=0.1,
            retention_schedule="linear_decay",
            retention_final_scale=0.12,
            anchor_ce_final_scale=0.3,
            calibration_mode="balanced_affine",
            calibration_old_budget=64,
            calibration_new_budget=1024,
            calibration_epochs=25,
            calibration_learning_rate=2e-2,
        ),
        MethodConfig(
            name="spma_old_geometry_balanced_512",
            description="Preserve only the old-task manifold geometry on anchors while leaving new-task features unconstrained.",
            lambda_kd=0.9,
            lambda_anchor_ce=1.0,
            lambda_geo=0.1,
            lambda_smooth=0.05,
            lambda_reg=0.0015,
            lambda_chart=0.35,
            temperature=2.0,
            chart_temperature=0.75,
            support_metric="factor_nll",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="full_ce",
            retention_schedule="linear_decay",
            retention_final_scale=0.1,
            anchor_ce_final_scale=0.25,
            calibration_mode="balanced_affine",
            calibration_old_budget=128,
            calibration_new_budget=768,
            calibration_epochs=25,
            calibration_learning_rate=2e-2,
        ),
        MethodConfig(
            name="spma_old_geometry_hi_new_512",
            description="Higher-new-accuracy old-geometry preservation with weaker anchor retention and no new-sample manifold constraints.",
            lambda_kd=0.5,
            lambda_anchor_ce=0.75,
            lambda_geo=0.05,
            lambda_smooth=0.02,
            lambda_reg=0.001,
            lambda_chart=0.2,
            temperature=2.0,
            chart_temperature=0.75,
            support_metric="factor_nll",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=512,
            new_task_loss_mode="full_ce",
            retention_schedule="linear_decay",
            retention_final_scale=0.03,
            anchor_ce_final_scale=0.15,
            calibration_mode="balanced_affine",
            calibration_old_budget=64,
            calibration_new_budget=1024,
            calibration_epochs=25,
            calibration_learning_rate=2e-2,
        ),
        MethodConfig(
            name="spma_class_conditional",
            description="SPMA with class-conditional support memories and class-aware support loss.",
            lambda_kd=1.0,
            lambda_geo=0.5,
            lambda_support=0.1,
            lambda_reg=0.005,
            support_conditioning="class_conditional",
        ),
        MethodConfig(
            name="spma_gmm",
            description="SPMA with a density model over old latents using a Gaussian mixture support score.",
            lambda_kd=1.0,
            lambda_geo=0.5,
            lambda_support=0.02,
            lambda_reg=0.005,
            support_metric="gmm_nll",
            manifold_builder="gmm",
        ),
        MethodConfig(
            name="spma_class_conditional_gmm",
            description="SPMA with class-conditional Gaussian-mixture support memories.",
            lambda_kd=1.0,
            lambda_geo=0.5,
            lambda_support=0.02,
            lambda_reg=0.005,
            support_metric="gmm_nll",
            support_conditioning="class_conditional",
            manifold_builder="gmm",
        ),
        MethodConfig(
            name="spma_lora",
            description="SPMA with LoRA adapters on the feature extractor and a trainable classifier head.",
            lambda_kd=1.0,
            lambda_geo=0.5,
            lambda_support=0.1,
            lambda_reg=0.005,
            trainable_mode="lora",
        ),
        MethodConfig(
            name="spma_hdbscan",
            description="SPMA with an HDBSCAN support memory for larger and less spherical latent structure.",
            lambda_kd=1.0,
            lambda_geo=0.5,
            lambda_support=0.1,
            lambda_reg=0.005,
            manifold_builder="hdbscan",
        ),
        MethodConfig(
            name="spma_hdbscan_novel",
            description="Novel-class SPMA with HDBSCAN memory and novelty-gated bounded support expansion.",
            lambda_kd=1.0,
            lambda_geo=0.5,
            lambda_support=0.05,
            lambda_reg=0.005,
            manifold_builder="hdbscan",
            support_loss_mode="bounded_expansion",
        ),
        MethodConfig(
            name="spma_hdbscan_sparse_smooth",
            description="Sparse-anchor SPMA with HDBSCAN memory, anchor CE, and local manifold smoothing.",
            lambda_kd=1.0,
            lambda_anchor_ce=1.0,
            lambda_geo=0.25,
            lambda_smooth=0.5,
            lambda_support=0.05,
            lambda_reg=0.005,
            manifold_builder="hdbscan",
            support_loss_mode="bounded_expansion",
            anchor_sampling_mode="cluster_stratified",
            anchor_total_budget_override=256,
        ),
        MethodConfig(
            name="spma_cc_gmm_lora",
            description="SPMA with class-conditional GMM support and LoRA feature updates.",
            lambda_kd=1.0,
            lambda_geo=0.5,
            lambda_support=0.02,
            lambda_reg=0.005,
            support_metric="gmm_nll",
            support_conditioning="class_conditional",
            manifold_builder="gmm",
            trainable_mode="lora",
        ),
    ]
    return base_methods


def build_ablation_methods(config: ExperimentConfig) -> list[MethodConfig]:
    base = next(method for method in build_baseline_methods(config) if method.name == "spma_full")
    return [
        replace(base, name="spma_low_lambda_kd", description="SPMA with weaker KD.", lambda_kd=0.25),
        replace(base, name="spma_high_lambda_kd", description="SPMA with stronger KD.", lambda_kd=2.0),
        replace(base, name="spma_low_lambda_geo", description="SPMA with weaker geometry preservation.", lambda_geo=0.1),
        replace(base, name="spma_high_lambda_geo", description="SPMA with stronger geometry preservation.", lambda_geo=2.0),
        replace(base, name="spma_low_lambda_support", description="SPMA with weaker support assimilation.", lambda_support=0.025),
        replace(base, name="spma_high_lambda_support", description="SPMA with stronger support assimilation.", lambda_support=0.5),
        replace(base, name="spma_small_anchor_buffer", description="SPMA with a smaller anchor buffer.", anchor_buffer_per_class_override=max(16, config.anchor_buffer_per_class // 2)),
        replace(base, name="spma_large_anchor_buffer", description="SPMA with a larger anchor buffer.", anchor_buffer_per_class_override=config.anchor_buffer_per_class + 32),
        replace(base, name="spma_k4", description="SPMA with fewer support clusters.", num_clusters_override=4),
        replace(base, name="spma_k12", description="SPMA with more support clusters.", num_clusters_override=12),
        replace(base, name="spma_euclidean_support", description="SPMA using Euclidean support distance.", support_metric="euclidean"),
        replace(base, name="spma_hdbscan_ablation", description="SPMA with HDBSCAN support components.", manifold_builder="hdbscan"),
        replace(
            base,
            name="spma_cc_kmeans",
            description="SPMA with class-conditional KMeans support memories.",
            support_conditioning="class_conditional",
        ),
        replace(
            base,
            name="spma_cc_gmm",
            description="SPMA with class-conditional GMM support memories.",
            lambda_support=0.02,
            support_conditioning="class_conditional",
            support_metric="gmm_nll",
            manifold_builder="gmm",
        ),
        replace(
            base,
            name="spma_lora_rank4",
            description="SPMA with lower-rank LoRA feature updates.",
            trainable_mode="lora",
            lora_rank_override=4,
            lora_alpha_override=4.0,
        ),
    ]
