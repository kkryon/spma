from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PAPER_DIR = ROOT / "paper"
FIGURE_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"

CURRENT_RUNS = {
    "cifar10_compatible_shift": ROOT / "outputs" / "spma" / "cifar10_compatible_current",
    "tiny_imagenet_compatible_shift": ROOT / "outputs" / "spma" / "notebook_tiny_imagenet",
}

METHOD_ORDER = [
    "plain_ft",
    "anchor_ce_only",
    "er_replay_512",
    "spma_old_geometry_balanced_512",
]

METHOD_LABELS = {
    "plain_ft": "Plain FT",
    "anchor_ce_only": "Anchor CE",
    "er_replay_512": "ER-512",
    "spma_old_geometry_balanced_512": "SPMA-OG",
}

METHOD_SHORT_LABELS = {
    "plain_ft": "Plain",
    "anchor_ce_only": "Anchor",
    "er_replay_512": "ER",
    "spma_old_geometry_balanced_512": "SPMA",
}

METHOD_COLORS = {
    "plain_ft": "#c44e52",
    "anchor_ce_only": "#4c72b0",
    "er_replay_512": "#55a868",
    "spma_old_geometry_balanced_512": "#1f9a8a",
}

REPRESENTATION_METHODS = [
    "plain_ft",
    "anchor_ce_only",
    "er_replay_512",
    "spma_old_geometry_balanced_512",
]

TASK_AWARE_BENCHMARKS = set()
_TASK_AWARE_RUN_CACHE: dict[tuple[str, int, str, str], tuple[float, float]] = {}
_TASK_AWARE_DATA_CACHE: dict[tuple[str, int, str], tuple[object, object, object, object, object]] = {}


def _ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _load_current_runs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for benchmark_name, run_dir in CURRENT_RUNS.items():
        csv_path = run_dir / "baseline_results.csv"
        frame = pd.read_csv(csv_path)
        frame["benchmark_name"] = benchmark_name
        frame["seed"] = 0
        frame["run_dir"] = str(run_dir)
        rows.extend(frame.to_dict(orient="records"))
    if not rows:
        raise FileNotFoundError("No curated baseline_results.csv files found for the current paper figures.")
    frame = pd.DataFrame(rows)
    numeric_columns = [
        "old_accuracy_after_ft",
        "new_accuracy_after_ft",
        "old_accuracy_before_ft",
        "forgetting",
        "cka",
        "pairwise_distance_correlation",
        "support_inside_empirical_after_ft",
        "support_distance_after_ft",
        "replay_anchor_size",
        "memory_anchor_size",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["harmonic_old_new"] = _harmonic(frame["old_accuracy_after_ft"], frame["new_accuracy_after_ft"])
    return frame


def _harmonic(old_values: pd.Series | np.ndarray, new_values: pd.Series | np.ndarray) -> np.ndarray:
    old_array = np.asarray(old_values, dtype=np.float64)
    new_array = np.asarray(new_values, dtype=np.float64)
    denominator = old_array + new_array
    return np.where(denominator > 0.0, 2.0 * old_array * new_array / denominator, 0.0)


def _task_aware_runtime(benchmark_name: str, seed: int, backbone_name: str):
    from spma.config import ExperimentConfig, build_default_config
    from spma.data import build_datasets, make_loader
    from spma.utils import configure_runtime, resolve_device, set_deterministic_seed

    cache_key = (benchmark_name, seed, backbone_name)
    cached = _TASK_AWARE_DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    config = build_default_config(ROOT / "data", ROOT / "paper" / ".tmp_out", ROOT / "paper" / ".tmp_ckpt")
    config = ExperimentConfig(
        **{
            **config.__dict__,
            "benchmark_name": benchmark_name,
            "seed": seed,
            "backbone_name": backbone_name,
            "num_workers": 0,
            "device": "auto",
            "enable_plots": False,
            "enable_ablations": False,
        }
    )
    set_deterministic_seed(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    configure_runtime(config, device)
    datasets = build_datasets(config)
    old_loader = make_loader(datasets.old_test, batch_size=config.eval_batch_size, shuffle=False, config=config)
    new_loader = make_loader(datasets.new_test, batch_size=config.eval_batch_size, shuffle=False, config=config)
    cached = (config, datasets, old_loader, new_loader, device)
    _TASK_AWARE_DATA_CACHE[cache_key] = cached
    return cached


def _compute_task_aware_metrics(run_dir: Path, benchmark_name: str, seed: int, method_name: str) -> tuple[float, float]:
    from spma.evaluate import evaluate_task_aware_accuracy
    from spma.models import build_model, wrap_with_dual_head

    cache_key = (benchmark_name, seed, method_name, str(run_dir))
    cached = _TASK_AWARE_RUN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    metrics_path = run_dir / f"{method_name}_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("classifier_mode") != "dual_head":
        cached = (float(metrics["old_accuracy_after_ft"]), float(metrics["new_accuracy_after_ft"]))
        _TASK_AWARE_RUN_CACHE[cache_key] = cached
        return cached

    config, datasets, old_loader, new_loader, device = _task_aware_runtime(
        benchmark_name=benchmark_name,
        seed=seed,
        backbone_name=str(metrics["backbone_name"]),
    )

    base_model = build_model(
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_classes=int(datasets.metadata["num_classes"]),
        input_shape=tuple(datasets.metadata["input_shape"]),
        backbone_name=str(metrics["backbone_name"]),
    )
    model = wrap_with_dual_head(
        base_model,
        old_classes=tuple(int(value) for value in datasets.metadata["old_classes"]),
        new_classes=tuple(int(value) for value in datasets.metadata["new_classes"]),
        reinitialize_new_head=True,
        new_head_bias_init=config.dual_head_new_head_bias_init,
    )
    state_dict = __import__("torch").load(metrics["checkpoint_path"], map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    old_task_aware = evaluate_task_aware_accuracy(model, old_loader, device, task="old", config=config)
    new_task_aware = evaluate_task_aware_accuracy(model, new_loader, device, task="new", config=config)
    model.cpu()
    cached = (float(old_task_aware), float(new_task_aware))
    _TASK_AWARE_RUN_CACHE[cache_key] = cached
    return cached


def _augment_with_task_aware_metrics(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    augmented = frame.copy()
    augmented["old_accuracy_after_ft_task_aware"] = augmented["old_accuracy_after_ft"]
    augmented["new_accuracy_after_ft_task_aware"] = augmented["new_accuracy_after_ft"]

    for row_index, row in augmented.iterrows():
        benchmark_name = str(row["benchmark_name"])
        if benchmark_name not in TASK_AWARE_BENCHMARKS:
            continue
        if "run_dir" in augmented.columns and isinstance(row.get("run_dir"), str):
            run_dir = Path(str(row["run_dir"]))
        else:
            run_dir = root / benchmark_name / f"seed_{int(row['seed'])}"
        metrics_path = run_dir / f"{row['method_name']}_metrics.json"
        if not metrics_path.exists():
            continue
        old_task_aware, new_task_aware = _compute_task_aware_metrics(
            run_dir=run_dir,
            benchmark_name=benchmark_name,
            seed=int(row["seed"]),
            method_name=str(row["method_name"]),
        )
        augmented.at[row_index, "old_accuracy_after_ft_task_aware"] = old_task_aware
        augmented.at[row_index, "new_accuracy_after_ft_task_aware"] = new_task_aware

    augmented["harmonic_old_new_task_aware"] = _harmonic(
        augmented["old_accuracy_after_ft_task_aware"],
        augmented["new_accuracy_after_ft_task_aware"],
    )
    return augmented


def _load_suite_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for csv_path in root.glob("*/*/baseline_results.csv"):
        benchmark_name = csv_path.parent.parent.name
        seed = int(csv_path.parent.name.replace("seed_", ""))
        frame = pd.read_csv(csv_path)
        frame["benchmark_name"] = benchmark_name
        frame["seed"] = seed
        rows.extend(frame.to_dict(orient="records"))
    if not rows:
        raise FileNotFoundError(f"No baseline_results.csv files found under {root}")
    frame = pd.DataFrame(rows)
    numeric_columns = [
        "old_accuracy_after_ft",
        "new_accuracy_after_ft",
        "old_accuracy_before_ft",
        "forgetting",
        "cka",
        "pairwise_distance_correlation",
        "support_inside_empirical_after_ft",
        "support_distance_after_ft",
        "replay_anchor_size",
        "memory_anchor_size",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["harmonic_old_new"] = _harmonic(frame["old_accuracy_after_ft"], frame["new_accuracy_after_ft"])
    return frame


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "old_accuracy_after_ft",
        "old_accuracy_after_ft_task_aware",
        "new_accuracy_after_ft",
        "new_accuracy_after_ft_task_aware",
        "forgetting",
        "cka",
        "pairwise_distance_correlation",
        "support_inside_empirical_after_ft",
        "support_distance_after_ft",
        "harmonic_old_new",
        "harmonic_old_new_task_aware",
    ]
    grouped = frame.groupby(["benchmark_name", "method_name"], as_index=False)
    mean_frame = grouped[metrics].mean().rename(columns={metric: f"{metric}_mean" for metric in metrics})
    std_frame = grouped[metrics].std(ddof=1).fillna(0.0).rename(columns={metric: f"{metric}_std" for metric in metrics})
    first_frame = grouped[["replay_anchor_size", "memory_anchor_size"]].first()
    merged = mean_frame.merge(std_frame, on=["benchmark_name", "method_name"]).merge(
        first_frame, on=["benchmark_name", "method_name"]
    )
    return merged


def _format_value(mean_value: float, std_value: float) -> str:
    if abs(std_value) < 1e-9:
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} $\\pm$ {std_value:.4f}"


def _format_best(mean_value: float, std_value: float, best_value: float) -> str:
    formatted = _format_value(mean_value, std_value)
    if abs(mean_value - best_value) < 1e-9:
        return f"\\textbf{{{formatted}}}"
    return formatted


def _method_tex(method_name: str) -> str:
    return METHOD_LABELS[method_name]


def _write_table(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_main_results_table(aggregate: pd.DataFrame) -> None:
    cifar = aggregate.loc[aggregate["benchmark_name"] == "cifar10_compatible_shift"].set_index("method_name")
    tiny = aggregate.loc[aggregate["benchmark_name"] == "tiny_imagenet_compatible_shift"].set_index("method_name")
    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Method & Replay & CIFAR10-Shift Old & CIFAR10-Shift New & CIFAR10-Shift Harm. & Tiny-Shift Old & Tiny-Shift New & Tiny-Shift Harm.\\\\",
        "\\midrule",
    ]
    for method_name in METHOD_ORDER:
        if method_name not in cifar.index or method_name not in tiny.index:
            continue
        cifar_row = cifar.loc[method_name]
        tiny_row = tiny.loc[method_name]
        cifar_old_best = float(cifar["old_accuracy_after_ft_mean"].max())
        cifar_new_best = float(cifar["new_accuracy_after_ft_mean"].max())
        cifar_harm_best = float(cifar["harmonic_old_new_mean"].max())
        tiny_old_best = float(tiny["old_accuracy_after_ft_mean"].max())
        tiny_new_best = float(tiny["new_accuracy_after_ft_mean"].max())
        tiny_harm_best = float(tiny["harmonic_old_new_mean"].max())
        replay_budget = int(cifar_row["replay_anchor_size"])
        lines.append(
            " & ".join(
                [
                    _method_tex(method_name),
                    str(replay_budget),
                    _format_best(float(cifar_row["old_accuracy_after_ft_mean"]), float(cifar_row["old_accuracy_after_ft_std"]), cifar_old_best),
                    _format_best(float(cifar_row["new_accuracy_after_ft_mean"]), float(cifar_row["new_accuracy_after_ft_std"]), cifar_new_best),
                    _format_best(float(cifar_row["harmonic_old_new_mean"]), float(cifar_row["harmonic_old_new_std"]), cifar_harm_best),
                    _format_best(float(tiny_row["old_accuracy_after_ft_mean"]), float(tiny_row["old_accuracy_after_ft_std"]), tiny_old_best),
                    _format_best(float(tiny_row["new_accuracy_after_ft_mean"]), float(tiny_row["new_accuracy_after_ft_std"]), tiny_new_best),
                    _format_best(float(tiny_row["harmonic_old_new_mean"]), float(tiny_row["harmonic_old_new_std"]), tiny_harm_best),
                ]
            )
            + "\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_table(TABLE_DIR / "main_results.tex", lines)


def build_representation_table(aggregate: pd.DataFrame) -> None:
    cifar = aggregate.loc[aggregate["benchmark_name"] == "cifar10_compatible_shift"].set_index("method_name")
    tiny = aggregate.loc[aggregate["benchmark_name"] == "tiny_imagenet_compatible_shift"].set_index("method_name")
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & CIFAR10-Shift CKA & CIFAR10-Shift Dist. Corr. & CIFAR10-Shift Support-In & Tiny CKA & Tiny Dist. Corr. & Tiny Support-In\\\\",
        "\\midrule",
    ]
    for method_name in REPRESENTATION_METHODS:
        if method_name not in cifar.index or method_name not in tiny.index:
            continue
        cifar_row = cifar.loc[method_name]
        tiny_row = tiny.loc[method_name]
        cifar_cka_best = float(cifar["cka_mean"].max())
        cifar_corr_best = float(cifar["pairwise_distance_correlation_mean"].max())
        cifar_support_best = float(cifar["support_inside_empirical_after_ft_mean"].max())
        tiny_cka_best = float(tiny["cka_mean"].max())
        tiny_corr_best = float(tiny["pairwise_distance_correlation_mean"].max())
        tiny_support_best = float(tiny["support_inside_empirical_after_ft_mean"].max())
        lines.append(
            " & ".join(
                [
                    _method_tex(method_name),
                    _format_best(float(cifar_row["cka_mean"]), float(cifar_row["cka_std"]), cifar_cka_best),
                    _format_best(
                        float(cifar_row["pairwise_distance_correlation_mean"]),
                        float(cifar_row["pairwise_distance_correlation_std"]),
                        cifar_corr_best,
                    ),
                    _format_best(
                        float(cifar_row["support_inside_empirical_after_ft_mean"]),
                        float(cifar_row["support_inside_empirical_after_ft_std"]),
                        cifar_support_best,
                    ),
                    _format_best(float(tiny_row["cka_mean"]), float(tiny_row["cka_std"]), tiny_cka_best),
                    _format_best(
                        float(tiny_row["pairwise_distance_correlation_mean"]),
                        float(tiny_row["pairwise_distance_correlation_std"]),
                        tiny_corr_best,
                    ),
                    _format_best(
                        float(tiny_row["support_inside_empirical_after_ft_mean"]),
                        float(tiny_row["support_inside_empirical_after_ft_std"]),
                        tiny_support_best,
                    ),
                ]
            )
            + "\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_table(TABLE_DIR / "representation_results.tex", lines)


def build_task_aware_results_table(aggregate: pd.DataFrame) -> None:
    cifar = aggregate.loc[aggregate["benchmark_name"] == "cifar10_compatible_shift"].set_index("method_name")
    tiny = aggregate.loc[aggregate["benchmark_name"] == "tiny_imagenet_compatible_shift"].set_index("method_name")
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & CIFAR10-Shift Old* & CIFAR10-Shift New* & CIFAR10-Shift Harm.* & Tiny-Shift Old* & Tiny-Shift New* & Tiny-Shift Harm.*\\\\",
        "\\midrule",
    ]
    for method_name in METHOD_ORDER:
        if method_name not in cifar.index or method_name not in tiny.index:
            continue
        cifar_row = cifar.loc[method_name]
        tiny_row = tiny.loc[method_name]
        lines.append(
            " & ".join(
                [
                    _method_tex(method_name),
                    _format_value(
                        float(cifar_row["old_accuracy_after_ft_task_aware_mean"]),
                        float(cifar_row["old_accuracy_after_ft_task_aware_std"]),
                    ),
                    _format_value(
                        float(cifar_row["new_accuracy_after_ft_task_aware_mean"]),
                        float(cifar_row["new_accuracy_after_ft_task_aware_std"]),
                    ),
                    _format_value(
                        float(cifar_row["harmonic_old_new_task_aware_mean"]),
                        float(cifar_row["harmonic_old_new_task_aware_std"]),
                    ),
                    _format_value(
                        float(tiny_row["old_accuracy_after_ft_task_aware_mean"]),
                        float(tiny_row["old_accuracy_after_ft_task_aware_std"]),
                    ),
                    _format_value(
                        float(tiny_row["new_accuracy_after_ft_task_aware_mean"]),
                        float(tiny_row["new_accuracy_after_ft_task_aware_std"]),
                    ),
                    _format_value(
                        float(tiny_row["harmonic_old_new_task_aware_mean"]),
                        float(tiny_row["harmonic_old_new_task_aware_std"]),
                    ),
                ]
            )
            + "\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_table(TABLE_DIR / "task_aware_results.tex", lines)


def build_shift_pilot_table(aggregate: pd.DataFrame) -> None:
    pilot = aggregate.loc[aggregate["benchmark_name"] == "tiny_imagenet_compatible_shift"].set_index("method_name")
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Replay & Old Before & Old After & New After & Harmonic Mean\\\\",
        "\\midrule",
    ]
    for method_name in METHOD_ORDER:
        if method_name not in pilot.index:
            continue
        row = pilot.loc[method_name]
        old_before_mean_column = pilot["old_accuracy_after_ft_mean"] + pilot["forgetting_mean"]
        old_before_best = float(old_before_mean_column.max())
        old_after_best = float(pilot["old_accuracy_after_ft_mean"].max())
        new_after_best = float(pilot["new_accuracy_after_ft_mean"].max())
        harmonic_best = float(pilot["harmonic_old_new_mean"].max())
        old_before_mean = float(row["old_accuracy_after_ft_mean"] + row["forgetting_mean"])
        old_before_std = float(np.sqrt(row["old_accuracy_after_ft_std"] ** 2 + row["forgetting_std"] ** 2))
        lines.append(
            " & ".join(
                [
                    _method_tex(method_name),
                    str(int(row["replay_anchor_size"])),
                    _format_best(old_before_mean, old_before_std, old_before_best),
                    _format_best(float(row["old_accuracy_after_ft_mean"]), float(row["old_accuracy_after_ft_std"]), old_after_best),
                    _format_best(float(row["new_accuracy_after_ft_mean"]), float(row["new_accuracy_after_ft_std"]), new_after_best),
                    _format_best(float(row["harmonic_old_new_mean"]), float(row["harmonic_old_new_std"]), harmonic_best),
                ]
            )
            + "\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_table(TABLE_DIR / "compatible_shift_results.tex", lines)


def build_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        patch = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.5,
            facecolor=color,
            edgecolor="#1f2937",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=11.5, color="#111827")

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.6, color="#374151"))

    box(0.06, 0.60, 0.23, 0.22, "Frozen teacher\n$f_{\\theta_0}$", "#dbeafe")
    box(0.38, 0.60, 0.23, 0.22, "Teacher anchor features\n$z_0 = h_{\\theta_0}(x)$", "#e0f2fe")
    box(0.70, 0.60, 0.24, 0.22, "Local chart memory\n$\\{\\mu_k, U_k, \\sigma_k^2\\}$", "#dcfce7")
    box(0.06, 0.16, 0.23, 0.22, "New-task batch\n$D_{\\mathrm{new}}$", "#fee2e2")
    box(0.38, 0.16, 0.23, 0.22, "Student features\n$z = h_{\\theta}(x)$", "#fae8ff")
    box(0.70, 0.13, 0.24, 0.28, "Fine-tuning objective\nCE(new) + CE(anchor)\n+ KD + geometry + chart", "#fef3c7")

    arrow(0.29, 0.71, 0.38, 0.71)
    arrow(0.61, 0.71, 0.70, 0.71)
    arrow(0.29, 0.27, 0.38, 0.27)
    arrow(0.61, 0.27, 0.70, 0.27)
    arrow(0.50, 0.60, 0.50, 0.38)
    arrow(0.82, 0.60, 0.82, 0.41)

    ax.text(0.06, 0.95, "Shared-manifold continuation", fontsize=13, fontweight="bold", color="#111827")
    ax.text(
        0.06,
        0.90,
        "Preserve old chart structure while adapting the student to the new task",
        fontsize=11.5,
        color="#111827",
    )
    ax.text(0.655, 0.76, "fit charts", fontsize=9.5, color="#374151")
    ax.text(0.655, 0.32, "optimize", fontsize=9.5, color="#374151")
    ax.text(0.50, 0.52, "KD + geometry", fontsize=9.5, color="#374151", ha="center")
    ax.text(0.82, 0.54, "chart signal", fontsize=9.5, color="#374151", ha="center")

    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURE_DIR / "method_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURE_DIR / "method_overview.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_tradeoff_figure(aggregate: pd.DataFrame) -> None:
    benchmarks = ["cifar10_compatible_shift", "tiny_imagenet_compatible_shift"]
    titles = {
        "cifar10_compatible_shift": "CIFAR10 compatible shift",
        "tiny_imagenet_compatible_shift": "Tiny-ImageNet compatible shift",
    }
    label_offsets = {
        "cifar10_compatible_shift": {
            "plain_ft": (10, -10),
            "anchor_ce_only": (10, 8),
            "er_replay_512": (10, -2),
            "spma_old_geometry_balanced_512": (10, 8),
        },
        "tiny_imagenet_compatible_shift": {
            "plain_ft": (10, -10),
            "anchor_ce_only": (10, -8),
            "er_replay_512": (10, 4),
            "spma_old_geometry_balanced_512": (10, 8),
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.9), sharex=False, sharey=False)
    for ax, benchmark_name in zip(axes, benchmarks, strict=True):
        bench = aggregate.loc[aggregate["benchmark_name"] == benchmark_name].copy()
        bench = bench.loc[bench["method_name"].isin(METHOD_ORDER)]
        for _, row in bench.iterrows():
            method_name = str(row["method_name"])
            x = float(row["new_accuracy_after_ft_mean"])
            y = float(row["old_accuracy_after_ft_mean"])
            ax.scatter(
                x,
                y,
                s=220 if method_name == "spma_old_geometry_balanced_512" else 150,
                color=METHOD_COLORS[method_name],
                edgecolor="#111827",
                linewidth=1.2,
                marker="*" if method_name == "spma_old_geometry_balanced_512" else "o",
                zorder=3,
            )
            dx, dy = label_offsets[benchmark_name][method_name]
            ax.annotate(
                METHOD_LABELS[method_name],
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.5,
                color="#111827",
                ha="left" if dx >= 0 else "right",
                va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8),
            )
        ax.set_title(titles[benchmark_name], fontsize=11)
        ax.set_xlabel("New-task accuracy")
        ax.set_ylabel("Old-task accuracy")
        ax.margins(x=0.12, y=0.14)
        ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)
    fig.savefig(FIGURE_DIR / "novel_tradeoff.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURE_DIR / "novel_tradeoff.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_representation_figure(aggregate: pd.DataFrame) -> None:
    metrics = [
        ("cka_mean", "CKA"),
        ("pairwise_distance_correlation_mean", "Pairwise corr."),
        ("support_inside_empirical_after_ft_mean", "Support-in"),
    ]
    benchmarks = ["cifar10_compatible_shift", "tiny_imagenet_compatible_shift"]
    titles = {
        "cifar10_compatible_shift": "CIFAR10 shift",
        "tiny_imagenet_compatible_shift": "Tiny-ImageNet shift",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.8), sharey=True)
    for row_index, benchmark_name in enumerate(benchmarks):
        bench = aggregate.loc[aggregate["benchmark_name"] == benchmark_name].set_index("method_name")
        methods = [method_name for method_name in REPRESENTATION_METHODS if method_name in bench.index]
        for column_index, (metric_name, metric_title) in enumerate(metrics):
            ax = axes[row_index, column_index]
            values = [float(bench.loc[method_name, metric_name]) for method_name in methods]
            errors = [float(bench.loc[method_name, metric_name.replace("_mean", "_std")]) for method_name in methods]
            colors = [METHOD_COLORS[method_name] for method_name in methods]
            bars = ax.bar(np.arange(len(methods)), values, yerr=errors, color=colors, edgecolor="#111827", linewidth=1.0)
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels([METHOD_SHORT_LABELS[method_name] for method_name in methods], rotation=0)
            ax.set_title(f"{titles[benchmark_name]}: {metric_title}", fontsize=11)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, axis="y", alpha=0.25)
            ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=8)
            if column_index == 0:
                ax.set_ylabel("Score")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURE_DIR / "representation_summary.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURE_DIR / "representation_summary.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_summary_files() -> None:
    current_rows = _load_current_runs()
    current_rows = _augment_with_task_aware_metrics(current_rows, ROOT / "outputs" / "spma")
    aggregate = _aggregate(current_rows)
    aggregate.to_csv(TABLE_DIR / "aggregate_summary.csv", index=False)

    build_main_results_table(aggregate)
    build_representation_table(aggregate)
    build_shift_pilot_table(aggregate)
    build_method_overview()
    build_tradeoff_figure(aggregate)
    build_representation_figure(aggregate)


def main() -> None:
    _ensure_dirs()
    build_summary_files()
    print(f"Wrote SPMA paper assets to {FIGURE_DIR} and {TABLE_DIR}")


if __name__ == "__main__":
    main()
