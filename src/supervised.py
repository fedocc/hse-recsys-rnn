from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset

from .models import RecurrentEncoder
from .pooling import pool_hidden_states


class PrefixLabelDataset(Dataset):
    def __init__(
        self,
        user_ids: Sequence[str],
        sequences: Sequence[Sequence[int]],
        labels: Sequence[int | float],
        *,
        prefix_len: int,
    ):
        if not (len(user_ids) == len(sequences) == len(labels)):
            raise ValueError("user_ids, sequences and labels must have the same length.")

        self.user_ids = list(user_ids)
        self.prefix_len = prefix_len
        self.labels = [float(label) for label in labels]
        self.sequences = [list(seq[:prefix_len]) for seq in sequences]

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        length = min(len(seq), self.prefix_len)
        padded = seq + [0] * (self.prefix_len - len(seq))
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class SupervisedRNNClassifier(nn.Module):
    def __init__(
        self,
        encoder: RecurrentEncoder,
        *,
        hidden_dim: int,
        pooling: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        embedding_dim = hidden_dim * 2 if pooling == "last_mean" else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def embed(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encoder.encode_steps(x, lengths)
        return pool_hidden_states(hidden_states, lengths, strategy=self.pooling)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed(x, lengths)
        return self.classifier(embeddings).squeeze(1)


def binary_pos_weight(labels: Sequence[int | float]) -> float:
    positives = float(sum(labels))
    negatives = float(len(labels) - positives)
    if positives == 0:
        return 1.0
    return negatives / positives


def train_supervised_classifier(
    model: SupervisedRNNClassifier,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: float,
) -> list[float]:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    history: list[float] = []
    model.train()

    for _ in range(epochs):
        total_loss = 0.0
        total_examples = 0
        for x, lengths, labels in dataloader:
            x = x.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            total_examples += len(labels)
        history.append(total_loss / max(1, total_examples))
    return history


@torch.no_grad()
def evaluate_supervised_classifier(
    model: SupervisedRNNClassifier,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    criterion = nn.BCEWithLogitsLoss()
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    total_loss = 0.0
    total_examples = 0
    model.eval()

    for x, lengths, labels in dataloader:
        x = x.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        logits = model(x, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        total_examples += len(labels)
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())

    if total_examples == 0:
        return {"loss": 0.0, "roc_auc": 0.0, "pr_auc": 0.0, "num_users": 0}

    logits_np = torch.cat(logits_all).numpy()
    labels_np = torch.cat(labels_all).numpy().astype(int)
    probs = torch.sigmoid(torch.tensor(logits_np)).numpy()
    return {
        "loss": total_loss / total_examples,
        "roc_auc": float(roc_auc_score(labels_np, probs)),
        "pr_auc": float(average_precision_score(labels_np, probs)),
        "num_users": int(total_examples),
    }


@torch.no_grad()
def build_supervised_embeddings(
    model: SupervisedRNNClassifier,
    dataset: PrefixLabelDataset,
    *,
    batch_size: int,
    device: torch.device,
    user_col: str = "appmetrica_device_id",
) -> pl.DataFrame:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows: list[dict[str, object]] = []
    model.eval()
    offset = 0
    for x, lengths, _ in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        embeddings = model.embed(x, lengths).cpu()
        batch_user_ids = dataset.user_ids[offset : offset + len(embeddings)]
        offset += len(embeddings)
        for user_id, embedding in zip(batch_user_ids, embeddings, strict=True):
            row = {user_col: user_id}
            row.update({f"emb_{idx:03d}": float(value) for idx, value in enumerate(embedding.tolist())})
            rows.append(row)
    return pl.DataFrame(rows)
