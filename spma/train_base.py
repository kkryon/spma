from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW

from .config import ExperimentConfig
from .data import DatasetBundle, make_loader
from .evaluate import collect_features_and_logits, evaluate_accuracy
from .losses import cross_entropy_loss
from .utils import autocast_context, build_grad_scaler, maybe_channels_last_model, move_to_device, save_npz


def train_base_model(
    model: torch.nn.Module,
    datasets: DatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
    checkpoint_path: str | Path,
    embeddings_path: str | Path,
) -> dict[str, list[float] | float | str]:
    model = maybe_channels_last_model(model, config)
    train_loader = make_loader(datasets.old_train, batch_size=config.batch_size, shuffle=True, config=config)
    test_loader = make_loader(datasets.old_test, batch_size=config.eval_batch_size, shuffle=False, config=config)
    optimizer = AdamW(model.parameters(), lr=config.base_learning_rate, weight_decay=config.weight_decay)
    scaler = build_grad_scaler(config, device)

    history = {"train_loss": [], "old_test_accuracy": []}
    best_accuracy = -1.0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.base_epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0
        for inputs, labels in train_loader:
            inputs = move_to_device(inputs, device, config)
            labels = labels.to(device, non_blocking=bool(config.fast_gpu_mode))

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config, device):
                logits = model(inputs)
                loss = cross_entropy_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

        epoch_loss = running_loss / max(total_examples, 1)
        epoch_accuracy = evaluate_accuracy(model, test_loader, device, config)
        history["train_loss"].append(epoch_loss)
        history["old_test_accuracy"].append(epoch_accuracy)
        print(f"[base] epoch={epoch:02d} loss={epoch_loss:.4f} old_acc={epoch_accuracy:.4f}")

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    embedding_cache = collect_features_and_logits(model, datasets.old_train, config, device)
    save_npz(
        embeddings_path,
        features=embedding_cache.features.astype("float32"),
        logits=embedding_cache.logits.astype("float32"),
        labels=embedding_cache.labels.astype("int64"),
    )
    history["best_old_test_accuracy"] = best_accuracy
    history["checkpoint_path"] = str(checkpoint_path)
    history["embeddings_path"] = str(embeddings_path)
    return history
