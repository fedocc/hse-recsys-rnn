from __future__ import annotations

import polars as pl
import torch

from .data import PreparedSequences
from .models import RecurrentEncoder


@torch.no_grad()
def build_user_embeddings(
    model: RecurrentEncoder,
    prepared: PreparedSequences,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> pl.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []

    for offset in range(0, len(prepared.user_ids), batch_size):
        batch_user_ids = prepared.user_ids[offset : offset + batch_size]
        batch_sequences = prepared.user_sequences[offset : offset + batch_size]
        lengths = torch.tensor([min(len(seq), max_len) for seq in batch_sequences], dtype=torch.long)
        padded = []
        for seq in batch_sequences:
            trimmed = seq[-max_len:]
            padded.append(trimmed + [0] * (max_len - len(trimmed)))

        x = torch.tensor(padded, dtype=torch.long, device=device)
        embeddings = model.encode(x, lengths.to(device)).cpu()

        for user_id, embedding in zip(batch_user_ids, embeddings, strict=True):
            row = {"appmetrica_device_id": user_id}
            row.update({f"emb_{idx:03d}": float(value) for idx, value in enumerate(embedding.tolist())})
            rows.append(row)

    return pl.DataFrame(rows)
