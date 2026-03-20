from __future__ import annotations

import csv
import json
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset


def set_deterministic_seed(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic)
    except RuntimeError:
        pass


def configure_runtime(config, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)
    torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark and not config.deterministic)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if config.allow_tf32 else "highest")


def autocast_context(config, device: torch.device):
    if bool(config.enable_amp) and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_grad_scaler(config, device: torch.device):
    enabled = bool(config.enable_amp) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def move_to_device(inputs: torch.Tensor, device: torch.device, config) -> torch.Tensor:
    tensor = inputs.to(device, non_blocking=bool(getattr(config, "fast_gpu_mode", False)))
    if bool(getattr(config, "channels_last", False)) and tensor.ndim == 4:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def maybe_channels_last_model(model: torch.nn.Module, config) -> torch.nn.Module:
    if bool(getattr(config, "channels_last", False)):
        model = model.to(memory_format=torch.channels_last)
    return model


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def select_balanced_indices(labels: torch.Tensor, classes: tuple[int, ...], per_class: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    indices: list[torch.Tensor] = []
    for class_id in classes:
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if class_indices.numel() == 0:
            continue
        take = min(per_class, class_indices.numel())
        permutation = torch.randperm(class_indices.numel(), generator=generator)
        indices.append(class_indices[permutation[:take]])
    return torch.cat(indices, dim=0)


def dataset_labels(dataset: Dataset) -> torch.Tensor:
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1]
    if isinstance(dataset, Subset):
        base_labels = dataset_labels(dataset.dataset)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return base_labels[indices]
    if isinstance(dataset, ConcatDataset):
        return torch.cat([dataset_labels(component) for component in dataset.datasets], dim=0)
    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        return labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels, dtype=torch.long)
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        return targets if isinstance(targets, torch.Tensor) else torch.as_tensor(targets, dtype=torch.long)
    raise TypeError(f"Unsupported dataset type for label extraction: {type(dataset)!r}")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def save_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(data), handle, indent=2)


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
