"""
Inference routines and temperature scaling optimization.

Generates evaluation logits and post-processed probabilities over test datasets.
Includes temperature scaling implementation (Guo et al., 2017) to optimize probability calibration.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model, dataset, device, batch_size=32, temperature=1.0, num_workers=2):
    """Run model inference over dataset and return predictions array dictionary."""
    model = model.to(device)
    model.eval()

    is_cuda = (device != "cpu" and (isinstance(device, str) or device.type == "cuda"))

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=is_cuda
    )

    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()

    # Temperature scaling calculation
    scaled_logits = logits / float(temperature)
    probs = F.softmax(scaled_logits, dim=1).numpy()
    pos_prob = probs[:, 1]  # Class index 1 corresponds to Pneumonia

    return {
        "y_true": labels,
        "y_prob": pos_prob,
        "logits": logits.numpy(),
    }


def fit_temperature(logits, labels, max_iter=200, lr=0.01):
    """Fit optimal temperature scaling hyperparameter on validation logits via NLL optimization."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    # Parameterize log_T to guarantee positivity during optimization
    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_T)
        loss = criterion(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_T).item())
