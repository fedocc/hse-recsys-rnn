from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        event_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        rnn_type: str = "gru",
    ):
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size + 1, event_dim, padding_idx=0)
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(
                event_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                event_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f"Unsupported rnn_type={rnn_type!r}. Expected 'gru' or 'lstm'.")
        self.projection = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.size(1),
        )
        return self.projection(out)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        return hidden[-1]

    def encode_steps(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.size(1),
        )
        return out


class GRUEncoder(RecurrentEncoder):
    def __init__(self, vocab_size: int, event_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__(
            vocab_size=vocab_size,
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type="gru",
        )


class LSTMEncoder(RecurrentEncoder):
    def __init__(self, vocab_size: int, event_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__(
            vocab_size=vocab_size,
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type="lstm",
        )


RNNEncoder = GRUEncoder
