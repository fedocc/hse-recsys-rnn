from .data import (
    EXCLUDED_EVENTS,
    EXCLUDED_PREFIXES,
    EventWindowDataset,
    NextEventExample,
    PreparedSequences,
    TECH_EVENTS,
    build_event_vocabulary,
    build_next_event_examples,
    build_user_sequences,
    load_event_logs,
    load_clean_event_logs,
    load_prepared_sequences,
    preprocess_event_logs_to_parquet,
    load_sequences,
    prepared_sequences_to_frame,
    save_prepared_sequences,
    summarize_event_logs,
    summarize_sequence_lengths,
)
from .splits import (
    UserSplit,
    apply_user_split,
    build_user_split,
    describe_user_split,
    load_user_split,
    save_user_split,
    save_user_split_tables,
)

__all__ = [
    "EXCLUDED_EVENTS",
    "EXCLUDED_PREFIXES",
    "EventWindowDataset",
    "NextEventExample",
    "PreparedSequences",
    "TECH_EVENTS",
    "UserSplit",
    "apply_user_split",
    "build_event_vocabulary",
    "build_next_event_examples",
    "build_user_sequences",
    "build_user_split",
    "describe_user_split",
    "load_clean_event_logs",
    "load_event_logs",
    "load_prepared_sequences",
    "load_sequences",
    "load_user_split",
    "preprocess_event_logs_to_parquet",
    "prepared_sequences_to_frame",
    "save_prepared_sequences",
    "save_user_split",
    "save_user_split_tables",
    "summarize_event_logs",
    "summarize_sequence_lengths",
]

try:
    from .baselines import Markov1Baseline, MostPopularBaseline, evaluate_baseline
    from .eval import evaluate_next_event_model, ranking_metrics_from_logits
    from .export import build_user_embeddings
    from .models import GRUEncoder, LSTMEncoder, RNNEncoder, RecurrentEncoder
    from .pooling import build_user_embeddings_with_pooling, pool_hidden_states
    from .train import choose_device, seed_everything, train_model
except ModuleNotFoundError:
    pass
else:
    __all__.extend(
        [
            "GRUEncoder",
            "LSTMEncoder",
            "Markov1Baseline",
            "MostPopularBaseline",
            "RNNEncoder",
            "RecurrentEncoder",
            "build_user_embeddings",
            "build_user_embeddings_with_pooling",
            "choose_device",
            "evaluate_baseline",
            "evaluate_next_event_model",
            "pool_hidden_states",
            "ranking_metrics_from_logits",
            "seed_everything",
            "train_model",
        ]
    )
