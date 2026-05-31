from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import NextEventExample
from .models import RecurrentEncoder


def ranking_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
    ignore_index: int = 0,
) -> dict[str, float]:
    if logits.ndim != 2:
        raise ValueError("Expected logits with shape [num_examples, vocab_size].")
    if targets.ndim != 1:
        raise ValueError("Expected targets with shape [num_examples].")
    if logits.size(0) != targets.size(0):
        raise ValueError("Logits and targets must have the same number of examples.")

    mask = targets.ne(ignore_index)
    if not torch.any(mask):
        return {f"hit@{k}": 0.0 for k in ks} | {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0}

    filtered_logits = logits[mask]
    filtered_targets = targets[mask]
    target_scores = filtered_logits.gather(1, filtered_targets.unsqueeze(1)).squeeze(1)
    ranks = filtered_logits.gt(target_scores.unsqueeze(1)).sum(dim=1) + 1

    metrics = {"mrr": ranks.float().reciprocal().mean().item()}
    for k in ks:
        hit_value = ranks.le(k).float().mean().item()
        metrics[f"hit@{k}"] = hit_value
        metrics[f"recall@{k}"] = hit_value
    return metrics


@torch.no_grad()
def evaluate_next_event_model(
    model: RecurrentEncoder,
    dataloader: DataLoader,
    device: torch.device,
    ks: Sequence[int] = (1, 5, 10),
    ignore_index: int = 0,
) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    metric_sums = {"mrr": 0.0, **{f"hit@{k}": 0.0 for k in ks}, **{f"recall@{k}": 0.0 for k in ks}}

    for batch, lengths in dataloader:
        batch = batch.to(device)
        input_lengths = (lengths - 1).clamp_min(1).to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs, input_lengths)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)

        loss = criterion(flat_logits, flat_targets)
        non_pad = flat_targets.ne(ignore_index)
        num_examples = int(non_pad.sum().item())
        if num_examples == 0:
            continue

        batch_metrics = ranking_metrics_from_logits(flat_logits, flat_targets, ks=ks, ignore_index=ignore_index)
        total_loss += loss.item() * num_examples
        total_tokens += num_examples
        for key, value in batch_metrics.items():
            metric_sums[key] += value * num_examples

    if total_tokens == 0:
        return {"loss": 0.0, **{key: 0.0 for key in metric_sums}}

    return {
        "loss": total_loss / total_tokens,
        **{key: value / total_tokens for key, value in metric_sums.items()},
    }


@torch.no_grad()
def evaluate_next_event_model_on_examples(
    model: RecurrentEncoder,
    examples: Sequence[NextEventExample],
    device: torch.device,
    ks: Sequence[int] = (1, 5, 10),
    batch_size: int = 4096,
    ignore_index: int = 0,
) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    model.eval()
    total_loss = 0.0
    total_examples = 0
    metric_sums = {"mrr": 0.0, **{f"hit@{k}": 0.0 for k in ks}, **{f"recall@{k}": 0.0 for k in ks}}

    for offset in range(0, len(examples), batch_size):
        batch_examples = examples[offset : offset + batch_size]
        if not batch_examples:
            continue

        lengths = torch.tensor([len(item.prefix) for item in batch_examples], dtype=torch.long, device=device)
        max_len = int(lengths.max().item())
        padded = [item.prefix + [0] * (max_len - len(item.prefix)) for item in batch_examples]
        inputs = torch.tensor(padded, dtype=torch.long, device=device)
        targets = torch.tensor([item.target for item in batch_examples], dtype=torch.long, device=device)

        logits_by_step = model(inputs, lengths)
        last_indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, logits_by_step.size(-1))
        logits = logits_by_step.gather(1, last_indices).squeeze(1)
        loss = criterion(logits, targets)
        batch_metrics = ranking_metrics_from_logits(logits, targets, ks=ks, ignore_index=ignore_index)

        num_examples = len(batch_examples)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        for key, value in batch_metrics.items():
            metric_sums[key] += value * num_examples

    if total_examples == 0:
        return {"loss": 0.0, **{key: 0.0 for key in metric_sums}}

    return {
        "loss": total_loss / total_examples,
        **{key: value / total_examples for key, value in metric_sums.items()},
    }
