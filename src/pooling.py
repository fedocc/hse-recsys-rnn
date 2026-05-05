from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import torch

from .data import PreparedSequences
from .models import RecurrentEncoder


SUPPORTED_POOLING = {"last", "mean", "max", "last_mean"}


def pool_hidden_states(
    hidden_states: torch.Tensor,
    lengths: torch.Tensor,
    *,
    strategy: str,
) -> torch.Tensor:
    if strategy not in SUPPORTED_POOLING:
        raise ValueError(f"Unsupported pooling={strategy!r}. Expected one of {sorted(SUPPORTED_POOLING)}.")

    device = hidden_states.device
    batch_size, max_len, _ = hidden_states.shape
    lengths = lengths.to(device).clamp_min(1)
    last_indices = (lengths - 1).view(batch_size, 1, 1).expand(-1, 1, hidden_states.size(-1))
    last = hidden_states.gather(1, last_indices).squeeze(1)

    if strategy == "last":
        return last

    mask = torch.arange(max_len, device=device).view(1, -1) < lengths.view(-1, 1)
    masked = hidden_states * mask.unsqueeze(-1)

    if strategy == "mean":
        return masked.sum(dim=1) / lengths.view(-1, 1)

    if strategy == "max":
        return hidden_states.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values

    mean = masked.sum(dim=1) / lengths.view(-1, 1)
    return torch.cat([last, mean], dim=1)


def pad_sequences_for_prefix(
    sequences: Sequence[Sequence[int]],
    *,
    max_len: int,
    take_last: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([min(len(seq), max_len) for seq in sequences], dtype=torch.long)
    padded: list[list[int]] = []
    for seq in sequences:
        trimmed = list(seq[-max_len:] if take_last else seq[:max_len])
        padded.append(trimmed + [0] * (max_len - len(trimmed)))
    return torch.tensor(padded, dtype=torch.long), lengths


@torch.no_grad()
def build_user_embeddings_with_pooling(
    model: RecurrentEncoder,
    prepared: PreparedSequences,
    *,
    max_len: int,
    batch_size: int,
    device: torch.device,
    pooling: str = "last",
    take_last: bool = True,
    user_col: str = "appmetrica_device_id",
) -> pl.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []

    for offset in range(0, len(prepared.user_ids), batch_size):
        batch_user_ids = prepared.user_ids[offset : offset + batch_size]
        batch_sequences = prepared.user_sequences[offset : offset + batch_size]
        x_cpu, lengths = pad_sequences_for_prefix(batch_sequences, max_len=max_len, take_last=take_last)

        x = x_cpu.to(device)
        hidden_states = model.encode_steps(x, lengths.to(device))
        embeddings = pool_hidden_states(hidden_states, lengths.to(device), strategy=pooling).cpu()

        for user_id, embedding in zip(batch_user_ids, embeddings, strict=True):
            row = {user_col: user_id}
            row.update({f"emb_{idx:03d}": float(value) for idx, value in enumerate(embedding.tolist())})
            rows.append(row)

    return pl.DataFrame(rows)
