"""
Core PyTorch training and validation loop routines.

Executes complete training process with mixed precision support and validation-loss early stopping.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_class_weights(labels, num_classes=2):
    """Calculate inverse-frequency class weights for loss weighting."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one(model, train_ds, val_ds, device, *,
              lr, weight_decay, max_epochs, patience,
              batch_size, mixed_precision, num_workers=2, verbose=True):
    """Train PyTorch model with early stopping on validation loss."""
    model = model.to(device)

    is_cuda = (device != "cpu" and (isinstance(device, str) or device.type == "cuda"))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=is_cuda
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=is_cuda
    )

    class_weights = compute_class_weights(train_ds.labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = bool(mixed_precision) and is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(max_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        n_train_samples = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_curr = imgs.size(0)
            train_loss += loss.item() * batch_size_curr
            n_train_samples += batch_size_curr

        train_loss /= max(n_train_samples, 1)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

                batch_size_curr = imgs.size(0)
                val_loss += loss.item() * batch_size_curr
                n_val_samples += batch_size_curr

        val_loss /= max(n_val_samples, 1)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if verbose:
            improved_flag = "  *" if val_loss < best_val_loss else ""
            print(f"    Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{improved_flag}")

        # --- Early Stopping Evaluation ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"    Early stopping triggered at epoch {epoch} (Best epoch: {best_epoch}, Val Loss: {best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    return model, {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }

