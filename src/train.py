from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import RecurrentEncoder


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: RecurrentEncoder,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> list[float]:
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_tokens = 0

        for batch, lengths in dataloader:
            batch = batch.to(device)
            input_lengths = (lengths - 1).clamp_min(1).to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits = model(inputs, input_lengths)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            non_pad_tokens = targets.ne(0).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

        mean_loss = total_loss / max(1, total_tokens)
        history.append(mean_loss)
        print(f"epoch={epoch:02d} loss={mean_loss:.4f}")

    return history
