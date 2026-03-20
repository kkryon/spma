from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import ExperimentConfig
from .utils import select_balanced_indices


@dataclass(frozen=True)
class DatasetBundle:
    old_train: Dataset
    old_train_eval: Dataset
    old_test: Dataset
    new_train: Dataset
    new_test: Dataset
    anchor_memory: Dataset
    anchor_eval: Dataset
    viz_old: Dataset
    viz_new: Dataset
    metadata: dict[str, object] | None = None


class LabeledImageDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: torch.Tensor,
        transform,
    ) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.labels = labels.clone().long()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = int(self.labels[index].item())
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label


class ShiftedLabeledImageDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: torch.Tensor,
        *,
        angle_deg: float,
        translate_x: int,
        translate_y: int,
        scale: float,
        noise_std: float,
        blur_kernel: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        seed: int,
        apply_augmentation: bool,
    ) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.labels = labels.clone().long()
        self.angle_deg = angle_deg
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.scale = scale
        self.noise_std = noise_std
        self.blur_kernel = _odd_kernel_size(blur_kernel)
        self.mean = mean
        self.std = std
        self.seed = seed
        self.apply_augmentation = apply_augmentation

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_shift(self, image_tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        shifted = TF.affine(
            image_tensor,
            angle=self.angle_deg,
            translate=[self.translate_x, self.translate_y],
            scale=self.scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        if self.blur_kernel > 1:
            shifted = TF.gaussian_blur(shifted, kernel_size=[self.blur_kernel, self.blur_kernel])
        if self.noise_std > 0.0:
            shifted = shifted + torch.randn(shifted.shape, generator=generator) * self.noise_std
        return shifted.clamp_(0.0, 1.0)

    def _apply_augmentation(self, image_tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if not self.apply_augmentation:
            return image_tensor
        if torch.rand((), generator=generator).item() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
        padding = 4
        image_tensor = TF.pad(image_tensor, [padding, padding, padding, padding], fill=0.0)
        max_offset = padding * 2 + 1
        top = int(torch.randint(0, max_offset, (1,), generator=generator).item())
        left = int(torch.randint(0, max_offset, (1,), generator=generator).item())
        return TF.crop(image_tensor, top, left, 64, 64)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = int(self.labels[index].item())
        generator = torch.Generator().manual_seed(self.seed + index)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = TF.to_tensor(image)
        tensor = self._apply_shift(tensor, generator)
        tensor = self._apply_augmentation(tensor, generator)
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        return tensor, label


def _load_mnist_tensors(root: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = MNIST(root=root, train=train, download=True)
    data = dataset.data.float().div(255.0).unsqueeze(1)
    labels = dataset.targets.long()
    return data, labels


def _load_cifar_tensors(root: str, dataset_name: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_cls = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    dataset = dataset_cls(root=root, train=train, download=True)
    data = torch.from_numpy(dataset.data).float().permute(0, 3, 1, 2).div(255.0)
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    return data, labels


def _filter_classes(data: torch.Tensor, labels: torch.Tensor, classes: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.isin(labels, torch.tensor(classes))
    return data[mask], labels[mask]


def _subset_tensor_dataset(data: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> TensorDataset:
    return TensorDataset(data[indices], labels[indices])


def _subset_dataset(dataset: Dataset, indices: torch.Tensor) -> Dataset:
    return Subset(dataset, indices.tolist())


def _odd_kernel_size(kernel_size: int) -> int:
    if kernel_size <= 1:
        return 1
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def _apply_shift_transform(
    images: torch.Tensor,
    angle_deg: float,
    translate_x: int,
    translate_y: int,
    scale: float,
    noise_std: float,
    blur_kernel: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    kernel_size = _odd_kernel_size(blur_kernel)
    transformed_images = torch.empty_like(images)
    for index, image in enumerate(images):
        transformed = TF.affine(
            image,
            angle=angle_deg,
            translate=[translate_x, translate_y],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        if kernel_size > 1:
            transformed = TF.gaussian_blur(transformed, kernel_size=[kernel_size, kernel_size])
        if noise_std > 0.0:
            transformed = transformed + torch.randn(transformed.shape, generator=generator) * noise_std
        transformed_images[index] = transformed.clamp_(0.0, 1.0)
    return transformed_images


def _build_class_split_bundle(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    old_classes: tuple[int, ...],
    new_classes: tuple[int, ...],
    config: ExperimentConfig,
    benchmark_name: str,
    num_classes: int,
) -> DatasetBundle:
    old_train_x, old_train_y = _filter_classes(train_images, train_labels, old_classes)
    old_test_x, old_test_y = _filter_classes(test_images, test_labels, old_classes)
    new_train_x, new_train_y = _filter_classes(train_images, train_labels, new_classes)
    new_test_x, new_test_y = _filter_classes(test_images, test_labels, new_classes)

    anchor_indices = select_balanced_indices(old_train_y, old_classes, config.anchor_buffer_per_class, config.seed + 11)
    anchor_eval_indices = select_balanced_indices(old_test_y, old_classes, config.anchor_eval_per_class, config.seed + 17)
    viz_old_indices = select_balanced_indices(old_test_y, old_classes, config.visualization_per_class, config.seed + 23)
    viz_new_indices = select_balanced_indices(new_test_y, new_classes, config.visualization_per_class, config.seed + 29)

    return DatasetBundle(
        old_train=TensorDataset(old_train_x, old_train_y),
        old_train_eval=TensorDataset(old_train_x, old_train_y),
        old_test=TensorDataset(old_test_x, old_test_y),
        new_train=TensorDataset(new_train_x, new_train_y),
        new_test=TensorDataset(new_test_x, new_test_y),
        anchor_memory=_subset_tensor_dataset(old_train_x, old_train_y, anchor_indices),
        anchor_eval=_subset_tensor_dataset(old_test_x, old_test_y, anchor_eval_indices),
        viz_old=_subset_tensor_dataset(old_test_x, old_test_y, viz_old_indices),
        viz_new=_subset_tensor_dataset(new_test_x, new_test_y, viz_new_indices),
        metadata={
            "benchmark_name": benchmark_name,
            "task_relation": "disjoint_classes",
            "old_classes": old_classes,
            "new_classes": new_classes,
            "input_shape": tuple(train_images.shape[1:]),
            "num_classes": num_classes,
        },
    )


def _build_shift_benchmark_datasets(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    config: ExperimentConfig,
    benchmark_name: str,
    classes: tuple[int, ...],
) -> DatasetBundle:
    old_train_x, old_train_y = _filter_classes(train_images, train_labels, classes)
    old_test_x, old_test_y = _filter_classes(test_images, test_labels, classes)

    if benchmark_name in {"mnist_compatible_shift", "cifar10_compatible_shift"}:
        angle_deg = config.compatible_shift_rotation_deg
        translate_x = config.compatible_shift_translate_x
        translate_y = config.compatible_shift_translate_y
        scale = config.compatible_shift_scale
        noise_std = config.compatible_shift_noise_std
        blur_kernel = config.compatible_shift_blur_kernel
    elif benchmark_name in {"mnist_stress_shift", "cifar10_stress_shift"}:
        angle_deg = config.stress_shift_rotation_deg
        translate_x = config.stress_shift_translate_x
        translate_y = config.stress_shift_translate_y
        scale = config.stress_shift_scale
        noise_std = config.stress_shift_noise_std
        blur_kernel = config.stress_shift_blur_kernel
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}")

    new_train_x = _apply_shift_transform(
        old_train_x,
        angle_deg=angle_deg,
        translate_x=translate_x,
        translate_y=translate_y,
        scale=scale,
        noise_std=noise_std,
        blur_kernel=blur_kernel,
        seed=config.seed + 101,
    )
    new_test_x = _apply_shift_transform(
        old_test_x,
        angle_deg=angle_deg,
        translate_x=translate_x,
        translate_y=translate_y,
        scale=scale,
        noise_std=noise_std,
        blur_kernel=blur_kernel,
        seed=config.seed + 103,
    )
    new_train_y = old_train_y.clone()
    new_test_y = old_test_y.clone()

    anchor_indices = select_balanced_indices(old_train_y, classes, config.anchor_buffer_per_class, config.seed + 11)
    anchor_eval_indices = select_balanced_indices(old_test_y, classes, config.anchor_eval_per_class, config.seed + 17)
    viz_old_indices = select_balanced_indices(old_test_y, classes, config.visualization_per_class, config.seed + 23)
    viz_new_indices = select_balanced_indices(new_test_y, classes, config.visualization_per_class, config.seed + 29)

    return DatasetBundle(
        old_train=TensorDataset(old_train_x, old_train_y),
        old_train_eval=TensorDataset(old_train_x, old_train_y),
        old_test=TensorDataset(old_test_x, old_test_y),
        new_train=TensorDataset(new_train_x, new_train_y),
        new_test=TensorDataset(new_test_x, new_test_y),
        anchor_memory=_subset_tensor_dataset(old_train_x, old_train_y, anchor_indices),
        anchor_eval=_subset_tensor_dataset(old_test_x, old_test_y, anchor_eval_indices),
        viz_old=_subset_tensor_dataset(old_test_x, old_test_y, viz_old_indices),
        viz_new=_subset_tensor_dataset(new_test_x, new_test_y, viz_new_indices),
        metadata={
            "benchmark_name": benchmark_name,
            "task_relation": "same_classes_shifted_domain",
            "classes": classes,
            "angle_deg": angle_deg,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "scale": scale,
            "noise_std": noise_std,
            "blur_kernel": blur_kernel,
            "input_shape": tuple(train_images.shape[1:]),
            "num_classes": max(classes) + 1,
        },
    )


def _resolve_tiny_imagenet_root(root: Path) -> Path:
    if (root / "train").is_dir() and (root / "val").is_dir():
        return root
    nested_root = root / "tiny-imagenet-200"
    if (nested_root / "train").is_dir() and (nested_root / "val").is_dir():
        return nested_root
    raise FileNotFoundError(f"Tiny-ImageNet not found under {root}.")


def _download_tiny_imagenet(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "tiny-imagenet-200.zip"
    dataset_root = root / "tiny-imagenet-200"
    if dataset_root.is_dir():
        return dataset_root
    if not archive_path.exists():
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        print(f"[data] downloading Tiny-ImageNet from {url}")
        urllib.request.urlretrieve(url, archive_path)
    print(f"[data] extracting {archive_path}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(root)
    return dataset_root


def _read_tiny_imagenet_wnids(dataset_root: Path) -> list[str]:
    wnids_path = dataset_root / "wnids.txt"
    if wnids_path.exists():
        return [line.strip() for line in wnids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(path.name for path in (dataset_root / "train").iterdir() if path.is_dir())


def _collect_tiny_imagenet_train(dataset_root: Path, class_to_idx: dict[str, int]) -> dict[int, list[Path]]:
    train_root = dataset_root / "train"
    grouped: dict[int, list[Path]] = {index: [] for index in class_to_idx.values()}
    for class_name, class_index in class_to_idx.items():
        image_dir = train_root / class_name / "images"
        if not image_dir.is_dir():
            continue
        grouped[class_index] = sorted(path for path in image_dir.iterdir() if path.is_file())
    return grouped


def _collect_tiny_imagenet_val(dataset_root: Path, class_to_idx: dict[str, int]) -> dict[int, list[Path]]:
    val_root = dataset_root / "val"
    annotations_path = val_root / "val_annotations.txt"
    images_root = val_root / "images"
    grouped: dict[int, list[Path]] = {index: [] for index in class_to_idx.values()}
    if annotations_path.exists() and images_root.is_dir():
        for line in annotations_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            image_name, class_name = parts[0], parts[1]
            if class_name not in class_to_idx:
                continue
            image_path = images_root / image_name
            if image_path.is_file():
                grouped[class_to_idx[class_name]].append(image_path)
        return grouped
    for class_name, class_index in class_to_idx.items():
        image_dir = val_root / class_name
        if image_dir.is_dir():
            grouped[class_index] = sorted(path for path in image_dir.iterdir() if path.is_file())
    return grouped


def _build_tiny_imagenet_split(config: ExperimentConfig) -> DatasetBundle:
    base_root = Path(config.data_dir) / "tiny_imagenet"
    try:
        dataset_root = _resolve_tiny_imagenet_root(base_root)
    except FileNotFoundError:
        dataset_root = _download_tiny_imagenet(base_root)
        dataset_root = _resolve_tiny_imagenet_root(base_root)

    class_names = _read_tiny_imagenet_wnids(dataset_root)
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    train_groups = _collect_tiny_imagenet_train(dataset_root, class_to_idx)
    val_groups = _collect_tiny_imagenet_val(dataset_root, class_to_idx)
    old_classes = tuple(range(100))
    new_classes = tuple(range(100, 200))

    train_mean = (0.485, 0.456, 0.406)
    train_std = (0.229, 0.224, 0.225)
    train_transform = Compose(
        [
            RandomHorizontalFlip(),
            RandomCrop(64, padding=4),
            ToTensor(),
            Normalize(mean=train_mean, std=train_std),
        ]
    )
    eval_transform = Compose([ToTensor(), Normalize(mean=train_mean, std=train_std)])

    old_train_paths = [path for class_id in old_classes for path in train_groups.get(class_id, [])]
    old_train_labels = torch.tensor([class_id for class_id in old_classes for _ in train_groups.get(class_id, [])], dtype=torch.long)
    new_train_paths = [path for class_id in new_classes for path in train_groups.get(class_id, [])]
    new_train_labels = torch.tensor([class_id for class_id in new_classes for _ in train_groups.get(class_id, [])], dtype=torch.long)
    old_test_paths = [path for class_id in old_classes for path in val_groups.get(class_id, [])]
    old_test_labels = torch.tensor([class_id for class_id in old_classes for _ in val_groups.get(class_id, [])], dtype=torch.long)
    new_test_paths = [path for class_id in new_classes for path in val_groups.get(class_id, [])]
    new_test_labels = torch.tensor([class_id for class_id in new_classes for _ in val_groups.get(class_id, [])], dtype=torch.long)

    old_train = LabeledImageDataset(old_train_paths, old_train_labels, transform=train_transform)
    old_train_eval = LabeledImageDataset(old_train_paths, old_train_labels, transform=eval_transform)
    new_train = LabeledImageDataset(new_train_paths, new_train_labels, transform=train_transform)
    old_test = LabeledImageDataset(old_test_paths, old_test_labels, transform=eval_transform)
    new_test = LabeledImageDataset(new_test_paths, new_test_labels, transform=eval_transform)

    anchor_indices = select_balanced_indices(old_train_labels, old_classes, config.anchor_buffer_per_class, config.seed + 11)
    anchor_eval_indices = select_balanced_indices(old_test_labels, old_classes, config.anchor_eval_per_class, config.seed + 17)
    viz_old_indices = select_balanced_indices(old_test_labels, old_classes, config.visualization_per_class, config.seed + 23)
    viz_new_indices = select_balanced_indices(new_test_labels, new_classes, config.visualization_per_class, config.seed + 29)

    return DatasetBundle(
        old_train=old_train,
        old_train_eval=old_train_eval,
        old_test=old_test,
        new_train=new_train,
        new_test=new_test,
        anchor_memory=_subset_dataset(old_train_eval, anchor_indices),
        anchor_eval=_subset_dataset(old_test, anchor_eval_indices),
        viz_old=_subset_dataset(old_test, viz_old_indices),
        viz_new=_subset_dataset(new_test, viz_new_indices),
        metadata={
            "benchmark_name": "split_tiny_imagenet",
            "task_relation": "disjoint_classes",
            "old_classes": old_classes,
            "new_classes": new_classes,
            "class_names": class_names,
            "input_shape": (3, 64, 64),
            "num_classes": 200,
        },
    )


def _build_tiny_imagenet_shift_benchmark(config: ExperimentConfig, benchmark_name: str) -> DatasetBundle:
    base_root = Path(config.data_dir) / "tiny_imagenet"
    try:
        dataset_root = _resolve_tiny_imagenet_root(base_root)
    except FileNotFoundError:
        dataset_root = _download_tiny_imagenet(base_root)
        dataset_root = _resolve_tiny_imagenet_root(base_root)

    class_names = _read_tiny_imagenet_wnids(dataset_root)
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    train_groups = _collect_tiny_imagenet_train(dataset_root, class_to_idx)
    val_groups = _collect_tiny_imagenet_val(dataset_root, class_to_idx)
    classes = tuple(range(len(class_names)))

    if benchmark_name == "tiny_imagenet_compatible_shift":
        angle_deg = config.compatible_shift_rotation_deg
        translate_x = config.compatible_shift_translate_x
        translate_y = config.compatible_shift_translate_y
        scale = config.compatible_shift_scale
        noise_std = config.compatible_shift_noise_std
        blur_kernel = config.compatible_shift_blur_kernel
    elif benchmark_name == "tiny_imagenet_stress_shift":
        angle_deg = config.stress_shift_rotation_deg
        translate_x = config.stress_shift_translate_x
        translate_y = config.stress_shift_translate_y
        scale = config.stress_shift_scale
        noise_std = config.stress_shift_noise_std
        blur_kernel = config.stress_shift_blur_kernel
    else:
        raise ValueError(f"Unsupported Tiny-ImageNet shift benchmark: {benchmark_name}")

    train_mean = (0.485, 0.456, 0.406)
    train_std = (0.229, 0.224, 0.225)
    train_transform = Compose(
        [
            RandomHorizontalFlip(),
            RandomCrop(64, padding=4),
            ToTensor(),
            Normalize(mean=train_mean, std=train_std),
        ]
    )
    eval_transform = Compose([ToTensor(), Normalize(mean=train_mean, std=train_std)])

    train_paths = [path for class_id in classes for path in train_groups.get(class_id, [])]
    train_labels = torch.tensor([class_id for class_id in classes for _ in train_groups.get(class_id, [])], dtype=torch.long)
    test_paths = [path for class_id in classes for path in val_groups.get(class_id, [])]
    test_labels = torch.tensor([class_id for class_id in classes for _ in val_groups.get(class_id, [])], dtype=torch.long)

    old_train = LabeledImageDataset(train_paths, train_labels, transform=train_transform)
    old_train_eval = LabeledImageDataset(train_paths, train_labels, transform=eval_transform)
    old_test = LabeledImageDataset(test_paths, test_labels, transform=eval_transform)
    new_train = ShiftedLabeledImageDataset(
        train_paths,
        train_labels,
        angle_deg=angle_deg,
        translate_x=translate_x,
        translate_y=translate_y,
        scale=scale,
        noise_std=noise_std,
        blur_kernel=blur_kernel,
        mean=train_mean,
        std=train_std,
        seed=config.seed + 101,
        apply_augmentation=True,
    )
    new_test = ShiftedLabeledImageDataset(
        test_paths,
        test_labels,
        angle_deg=angle_deg,
        translate_x=translate_x,
        translate_y=translate_y,
        scale=scale,
        noise_std=noise_std,
        blur_kernel=blur_kernel,
        mean=train_mean,
        std=train_std,
        seed=config.seed + 103,
        apply_augmentation=False,
    )

    anchor_indices = select_balanced_indices(train_labels, classes, config.anchor_buffer_per_class, config.seed + 11)
    anchor_eval_indices = select_balanced_indices(test_labels, classes, config.anchor_eval_per_class, config.seed + 17)
    viz_old_indices = select_balanced_indices(test_labels, classes, config.visualization_per_class, config.seed + 23)
    viz_new_indices = select_balanced_indices(test_labels, classes, config.visualization_per_class, config.seed + 29)

    return DatasetBundle(
        old_train=old_train,
        old_train_eval=old_train_eval,
        old_test=old_test,
        new_train=new_train,
        new_test=new_test,
        anchor_memory=_subset_dataset(old_train_eval, anchor_indices),
        anchor_eval=_subset_dataset(old_test, anchor_eval_indices),
        viz_old=_subset_dataset(old_test, viz_old_indices),
        viz_new=_subset_dataset(new_test, viz_new_indices),
        metadata={
            "benchmark_name": benchmark_name,
            "task_relation": "same_classes_shifted_domain",
            "classes": classes,
            "class_names": class_names,
            "angle_deg": angle_deg,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "scale": scale,
            "noise_std": noise_std,
            "blur_kernel": blur_kernel,
            "input_shape": (3, 64, 64),
            "num_classes": len(classes),
        },
    )


def build_datasets(config: ExperimentConfig) -> DatasetBundle:
    if config.benchmark_name == "split_mnist":
        train_images, train_labels = _load_mnist_tensors(str(config.data_dir), train=True)
        test_images, test_labels = _load_mnist_tensors(str(config.data_dir), train=False)
        return _build_class_split_bundle(
            train_images,
            train_labels,
            test_images,
            test_labels,
            old_classes=config.old_classes,
            new_classes=config.new_classes,
            config=config,
            benchmark_name="split_mnist",
            num_classes=10,
        )
    if config.benchmark_name in {"mnist_compatible_shift", "mnist_stress_shift"}:
        train_images, train_labels = _load_mnist_tensors(str(config.data_dir), train=True)
        test_images, test_labels = _load_mnist_tensors(str(config.data_dir), train=False)
        return _build_shift_benchmark_datasets(
            train_images,
            train_labels,
            test_images,
            test_labels,
            config,
            config.benchmark_name,
            classes=config.old_classes,
        )
    if config.benchmark_name == "split_cifar10":
        train_images, train_labels = _load_cifar_tensors(str(config.data_dir), "cifar10", train=True)
        test_images, test_labels = _load_cifar_tensors(str(config.data_dir), "cifar10", train=False)
        return _build_class_split_bundle(
            train_images,
            train_labels,
            test_images,
            test_labels,
            old_classes=(0, 1, 2, 3, 4),
            new_classes=(5, 6, 7, 8, 9),
            config=config,
            benchmark_name="split_cifar10",
            num_classes=10,
        )
    if config.benchmark_name == "split_cifar100":
        train_images, train_labels = _load_cifar_tensors(str(config.data_dir), "cifar100", train=True)
        test_images, test_labels = _load_cifar_tensors(str(config.data_dir), "cifar100", train=False)
        return _build_class_split_bundle(
            train_images,
            train_labels,
            test_images,
            test_labels,
            old_classes=tuple(range(50)),
            new_classes=tuple(range(50, 100)),
            config=config,
            benchmark_name="split_cifar100",
            num_classes=100,
        )
    if config.benchmark_name == "split_tiny_imagenet":
        return _build_tiny_imagenet_split(config)
    if config.benchmark_name in {"tiny_imagenet_compatible_shift", "tiny_imagenet_stress_shift"}:
        return _build_tiny_imagenet_shift_benchmark(config, config.benchmark_name)
    if config.benchmark_name in {"cifar10_compatible_shift", "cifar10_stress_shift"}:
        train_images, train_labels = _load_cifar_tensors(str(config.data_dir), "cifar10", train=True)
        test_images, test_labels = _load_cifar_tensors(str(config.data_dir), "cifar10", train=False)
        return _build_shift_benchmark_datasets(
            train_images,
            train_labels,
            test_images,
            test_labels,
            config,
            config.benchmark_name,
            classes=tuple(range(10)),
        )
    raise ValueError(f"Unsupported benchmark: {config.benchmark_name}")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, config: ExperimentConfig) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": config.device in {"cuda", "auto"},
        "drop_last": False,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(config.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(config.prefetch_factor)
    return DataLoader(**loader_kwargs)
