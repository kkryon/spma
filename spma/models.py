from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


def _pool_intermediate_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return F.adaptive_avg_pool2d(tensor, output_size=1).flatten(start_dim=1)
    if tensor.ndim > 2:
        return tensor.flatten(start_dim=1)
    return tensor


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)

        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        update = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return base + self.scaling * update


class SplitMNISTMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim > 2:
            inputs = inputs.view(inputs.size(0), -1)
        return self.feature_extractor(inputs)

    def forward_with_intermediates(
        self,
        inputs: torch.Tensor,
        layer_names: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        requested = set(layer_names or [])
        intermediates: dict[str, torch.Tensor] = {}
        if inputs.ndim > 2:
            inputs = inputs.view(inputs.size(0), -1)
        hidden1 = self.feature_extractor[1](self.feature_extractor[0](inputs))
        if "hidden1" in requested:
            intermediates["hidden1"] = hidden1
        penultimate = self.feature_extractor[3](self.feature_extractor[2](hidden1))
        if "penultimate" in requested:
            intermediates["penultimate"] = penultimate
        logits = self.classifier(penultimate)
        return logits, penultimate, intermediates

    def forward(self, inputs: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(inputs)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class SmallConvNet(nn.Module):
    def __init__(self, input_channels: int = 3, latent_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.conv_body = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_extractor = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        conv_features = self.conv_body(inputs)
        conv_features = conv_features.flatten(start_dim=1)
        return self.feature_extractor(conv_features)

    def forward_with_intermediates(
        self,
        inputs: torch.Tensor,
        layer_names: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        requested = set(layer_names or [])
        intermediates: dict[str, torch.Tensor] = {}
        x = self.conv_body[0](inputs)
        x = self.conv_body[1](x)
        x = self.conv_body[2](x)
        if "conv1" in requested:
            intermediates["conv1"] = _pool_intermediate_tensor(x)
        x = self.conv_body[3](x)
        x = self.conv_body[4](x)
        x = self.conv_body[5](x)
        if "conv2" in requested:
            intermediates["conv2"] = _pool_intermediate_tensor(x)
        x = self.conv_body[6](x)
        x = self.conv_body[7](x)
        x = self.conv_body[8](x)
        if "conv3" in requested:
            intermediates["conv3"] = _pool_intermediate_tensor(x)
        conv_features = x.flatten(start_dim=1)
        penultimate = self.feature_extractor(conv_features)
        if "penultimate" in requested:
            intermediates["penultimate"] = penultimate
        logits = self.classifier(penultimate)
        return logits, penultimate, intermediates

    def forward(self, inputs: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(inputs)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class ResNetCIFAR(nn.Module):
    def __init__(self, input_channels: int = 3, latent_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        backbone_features = self.backbone(inputs)
        return self.feature_extractor(backbone_features)

    def forward_with_intermediates(
        self,
        inputs: torch.Tensor,
        layer_names: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        requested = set(layer_names or [])
        intermediates: dict[str, torch.Tensor] = {}

        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        if "stem" in requested:
            intermediates["stem"] = _pool_intermediate_tensor(x)

        x = self.backbone.layer1(x)
        if "layer1" in requested:
            intermediates["layer1"] = _pool_intermediate_tensor(x)
        x = self.backbone.layer2(x)
        if "layer2" in requested:
            intermediates["layer2"] = _pool_intermediate_tensor(x)
        x = self.backbone.layer3(x)
        if "layer3" in requested:
            intermediates["layer3"] = _pool_intermediate_tensor(x)
        x = self.backbone.layer4(x)
        if "layer4" in requested:
            intermediates["layer4"] = _pool_intermediate_tensor(x)

        pooled = self.backbone.avgpool(x).flatten(start_dim=1)
        if "backbone" in requested:
            intermediates["backbone"] = pooled
        penultimate = self.feature_extractor(pooled)
        if "penultimate" in requested:
            intermediates["penultimate"] = penultimate
        logits = self.classifier(penultimate)
        return logits, penultimate, intermediates

    def forward(self, inputs: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(inputs)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class ResNetImageNet(nn.Module):
    def __init__(self, input_channels: int = 3, latent_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        if input_channels != 3:
            backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        backbone_features = self.backbone(inputs)
        return self.feature_extractor(backbone_features)

    def forward_with_intermediates(
        self,
        inputs: torch.Tensor,
        layer_names: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        requested = set(layer_names or [])
        intermediates: dict[str, torch.Tensor] = {}

        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        if "stem" in requested:
            intermediates["stem"] = _pool_intermediate_tensor(x)
        x = self.backbone.maxpool(x)
        if "maxpool" in requested:
            intermediates["maxpool"] = _pool_intermediate_tensor(x)

        x = self.backbone.layer1(x)
        if "layer1" in requested:
            intermediates["layer1"] = _pool_intermediate_tensor(x)
        x = self.backbone.layer2(x)
        if "layer2" in requested:
            intermediates["layer2"] = _pool_intermediate_tensor(x)
        x = self.backbone.layer3(x)
        if "layer3" in requested:
            intermediates["layer3"] = _pool_intermediate_tensor(x)
        x = self.backbone.layer4(x)
        if "layer4" in requested:
            intermediates["layer4"] = _pool_intermediate_tensor(x)

        pooled = self.backbone.avgpool(x).flatten(start_dim=1)
        if "backbone" in requested:
            intermediates["backbone"] = pooled
        penultimate = self.feature_extractor(pooled)
        if "penultimate" in requested:
            intermediates["penultimate"] = penultimate
        logits = self.classifier(penultimate)
        return logits, penultimate, intermediates

    def forward(self, inputs: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(inputs)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class DualHeadClassifierWrapper(nn.Module):
    def __init__(
        self,
        feature_model: nn.Module,
        old_classes: Sequence[int],
        new_classes: Sequence[int],
        reinitialize_new_head: bool = True,
        new_head_bias_init: float = -8.0,
    ) -> None:
        super().__init__()
        if not hasattr(feature_model, "classifier") or not isinstance(feature_model.classifier, nn.Linear):
            raise ValueError("Dual-head wrapping requires a base model with a linear classifier.")

        old_class_indices = tuple(int(index) for index in sorted(old_classes))
        new_class_indices = tuple(int(index) for index in sorted(new_classes))
        base_classifier: nn.Linear = feature_model.classifier
        latent_dim = base_classifier.in_features
        num_classes = base_classifier.out_features

        self.feature_model = feature_model
        self.old_head = nn.Linear(latent_dim, len(old_class_indices))
        self.new_head = nn.Linear(latent_dim, len(new_class_indices))
        self.num_classes = num_classes
        self.classifier_mode = "dual_head"
        self.parameter_drift_exempt_prefixes = (
            "new_head.",
            "old_logit_log_scale",
            "old_logit_bias",
            "new_logit_log_scale",
            "new_logit_bias",
        )

        self.old_logit_log_scale = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.old_logit_bias = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.new_logit_log_scale = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.new_logit_bias = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.register_buffer("old_class_index_tensor", torch.tensor(old_class_indices, dtype=torch.long), persistent=False)
        self.register_buffer("new_class_index_tensor", torch.tensor(new_class_indices, dtype=torch.long), persistent=False)

        with torch.no_grad():
            self.old_head.weight.copy_(base_classifier.weight[self.old_class_index_tensor])
            self.old_head.bias.copy_(base_classifier.bias[self.old_class_index_tensor])
            if reinitialize_new_head:
                self.new_head.weight.zero_()
                self.new_head.bias.fill_(new_head_bias_init)
            else:
                self.new_head.weight.copy_(base_classifier.weight[self.new_class_index_tensor])
                self.new_head.bias.copy_(base_classifier.bias[self.new_class_index_tensor])

        self.feature_model.classifier = nn.Identity()

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.feature_model.forward_features(inputs)

    def forward_with_intermediates(
        self,
        inputs: torch.Tensor,
        layer_names: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if hasattr(self.feature_model, "forward_with_intermediates"):
            _, features, intermediates = self.feature_model.forward_with_intermediates(inputs, layer_names=layer_names)
        else:
            features = self.feature_model.forward_features(inputs)
            intermediates = {}
        old_logits = self.old_head(features) * self.old_logit_log_scale.exp() + self.old_logit_bias
        new_logits = self.new_head(features) * self.new_logit_log_scale.exp() + self.new_logit_bias
        logits = old_logits.new_full((features.size(0), self.num_classes), -1e4)
        logits[:, self.old_class_index_tensor] = old_logits
        logits[:, self.new_class_index_tensor] = new_logits
        return logits, features, intermediates

    def forward(self, inputs: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(inputs)
        old_logits = self.old_head(features) * self.old_logit_log_scale.exp() + self.old_logit_bias
        new_logits = self.new_head(features) * self.new_logit_log_scale.exp() + self.new_logit_bias
        logits = old_logits.new_full((features.size(0), self.num_classes), -1e4)
        logits[:, self.old_class_index_tensor] = old_logits
        logits[:, self.new_class_index_tensor] = new_logits
        if return_features:
            return logits, features
        return logits


def build_model(
    hidden_dim: int,
    latent_dim: int,
    num_classes: int,
    input_shape: tuple[int, ...] = (1, 28, 28),
    backbone_name: str = "auto",
) -> nn.Module:
    if input_shape[0] == 1 and input_shape[1:] == (28, 28):
        return SplitMNISTMLP(hidden_dim=hidden_dim, latent_dim=latent_dim, num_classes=num_classes)
    resolved_backbone = backbone_name
    if resolved_backbone == "auto":
        resolved_backbone = "resnet18_imagenet" if max(input_shape[1:]) >= 64 else "resnet18_cifar"
    if resolved_backbone == "small_cnn":
        return SmallConvNet(input_channels=input_shape[0], latent_dim=max(latent_dim, 128), num_classes=num_classes)
    if resolved_backbone == "resnet18_cifar":
        return ResNetCIFAR(input_channels=input_shape[0], latent_dim=max(latent_dim, 128), num_classes=num_classes)
    if resolved_backbone == "resnet18_imagenet":
        return ResNetImageNet(input_channels=input_shape[0], latent_dim=max(latent_dim, 128), num_classes=num_classes)
    raise ValueError(f"Unsupported backbone_name: {backbone_name}")


def _replace_module_by_path(model: nn.Module, path: str, module: nn.Module) -> None:
    parent = model
    parts = path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def _iter_linear_module_paths(model: nn.Module) -> Iterable[str]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name


def enable_lora_finetuning(
    model: nn.Module,
    rank: int,
    alpha: float,
    keep_classifier_trainable: bool = True,
) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    excluded_paths = {"classifier", "old_head", "new_head"}
    linear_paths = [path for path in _iter_linear_module_paths(model) if path not in excluded_paths]
    for path in linear_paths:
        original = dict(model.named_modules())[path]
        _replace_module_by_path(model, path, LoRALinear(original, rank=rank, alpha=alpha))

    if keep_classifier_trainable and hasattr(model, "classifier"):
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)
    if hasattr(model, "old_head"):
        for parameter in model.old_head.parameters():
            parameter.requires_grad_(True)
    if hasattr(model, "new_head"):
        for parameter in model.new_head.parameters():
            parameter.requires_grad_(True)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def wrap_with_dual_head(
    model: nn.Module,
    old_classes: Sequence[int],
    new_classes: Sequence[int],
    reinitialize_new_head: bool = True,
    new_head_bias_init: float = -8.0,
) -> nn.Module:
    return DualHeadClassifierWrapper(
        feature_model=model,
        old_classes=old_classes,
        new_classes=new_classes,
        reinitialize_new_head=reinitialize_new_head,
        new_head_bias_init=new_head_bias_init,
    )


def forward_with_intermediates(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_names: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if hasattr(model, "forward_with_intermediates"):
        return model.forward_with_intermediates(inputs, layer_names=layer_names)
    outputs = model(inputs, return_features=True)
    logits, features = outputs
    intermediates = {"penultimate": features} if layer_names and "penultimate" in set(layer_names) else {}
    return logits, features, intermediates
