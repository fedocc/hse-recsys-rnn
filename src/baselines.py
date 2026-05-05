from __future__ import annotations

from collections.abc import Sequence

import torch

from .data import NextEventExample
from .eval import ranking_metrics_from_logits


class MostPopularBaseline:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.global_scores = torch.zeros(vocab_size + 1, dtype=torch.float32)

    def fit(self, user_sequences: Sequence[Sequence[int]]) -> "MostPopularBaseline":
        counts = torch.zeros(self.vocab_size + 1, dtype=torch.float32)
        for seq in user_sequences:
            for event_idx in seq[1:]:
                counts[event_idx] += 1
        self.global_scores = counts
        return self

    def predict_scores(self, prefixes: Sequence[Sequence[int]]) -> torch.Tensor:
        if not prefixes:
            return torch.empty((0, self.vocab_size + 1), dtype=torch.float32)
        return self.global_scores.unsqueeze(0).repeat(len(prefixes), 1)


class Markov1Baseline:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.transition_scores = torch.zeros((vocab_size + 1, vocab_size + 1), dtype=torch.float32)
        self.global_scores = torch.zeros(vocab_size + 1, dtype=torch.float32)

    def fit(self, user_sequences: Sequence[Sequence[int]]) -> "Markov1Baseline":
        transitions = torch.zeros((self.vocab_size + 1, self.vocab_size + 1), dtype=torch.float32)
        globals_ = torch.zeros(self.vocab_size + 1, dtype=torch.float32)

        for seq in user_sequences:
            for prev_idx, next_idx in zip(seq[:-1], seq[1:], strict=False):
                transitions[prev_idx, next_idx] += 1
                globals_[next_idx] += 1

        self.transition_scores = transitions
        self.global_scores = globals_
        return self

    def predict_scores(self, prefixes: Sequence[Sequence[int]]) -> torch.Tensor:
        rows = []
        for prefix in prefixes:
            last_event = prefix[-1] if prefix else 0
            scores = self.transition_scores[last_event]
            if float(scores.sum().item()) == 0.0:
                scores = self.global_scores
            rows.append(scores)

        if not rows:
            return torch.empty((0, self.vocab_size + 1), dtype=torch.float32)
        return torch.stack(rows)


def evaluate_baseline(
    baseline: MostPopularBaseline | Markov1Baseline,
    examples: Sequence[NextEventExample],
    ks: Sequence[int] = (1, 5, 10),
    batch_size: int = 4096,
) -> dict[str, float]:
    metric_sums = {"mrr": 0.0, **{f"hit@{k}": 0.0 for k in ks}, **{f"recall@{k}": 0.0 for k in ks}}
    total_examples = 0

    for offset in range(0, len(examples), batch_size):
        batch = examples[offset : offset + batch_size]
        prefixes = [item.prefix for item in batch]
        targets = torch.tensor([item.target for item in batch], dtype=torch.long)
        scores = baseline.predict_scores(prefixes)
        batch_metrics = ranking_metrics_from_logits(scores, targets, ks=ks, ignore_index=0)

        total_examples += len(batch)
        for key, value in batch_metrics.items():
            metric_sums[key] += value * len(batch)

    if total_examples == 0:
        return {key: 0.0 for key in metric_sums}
    return {key: value / total_examples for key, value in metric_sums.items()}
