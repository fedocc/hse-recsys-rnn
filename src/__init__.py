from .data import (
    EXCLUDED_EVENTS,
    EXCLUDED_PREFIXES,
    EventWindowDataset,
    LastEventWindowDataset,
    NextEventExample,
    PreparedSequences,
    TECH_EVENTS,
    build_event_vocabulary,
    build_last_event_examples,
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
    build_user_split_from_master_csv,
    describe_user_split,
    load_user_split,
    save_user_split,
    save_user_split_tables,
)

__all__ = [
    "EXCLUDED_EVENTS",
    "EXCLUDED_PREFIXES",
    "EventWindowDataset",
    "LastEventWindowDataset",
    "NextEventExample",
    "PreparedSequences",
    "TECH_EVENTS",
    "UserSplit",
    "apply_user_split",
    "build_event_vocabulary",
    "build_last_event_examples",
    "build_next_event_examples",
    "build_user_sequences",
    "build_user_split",
    "build_user_split_from_master_csv",
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
    from .controls import build_bag_of_events_features, build_random_embeddings, shuffle_user_prefixes, shuffle_user_sequences
    from .eval import evaluate_next_event_model, ranking_metrics_from_logits
    from .export import build_user_embeddings
    from .embedding_analysis import (
        add_embedding_norm,
        build_activity_bucket,
        cluster_profile,
        embedding_columns,
        embedding_matrix,
        fit_kmeans,
        pca_projection,
    )
    from .downstream import (
        build_base_meta,
        build_baseline_features,
        build_full_history_baseline_features,
        build_meta_for_full_history,
        build_meta_for_prefix,
        evaluate_xgboost_feature_set,
    )
    from .models import GRUEncoder, LSTMEncoder, RNNEncoder, RecurrentEncoder
    from .pooling import build_user_embeddings_with_pooling, pool_hidden_states
    from .supervised import (
        PrefixLabelDataset,
        SupervisedRNNClassifier,
        binary_pos_weight,
        build_supervised_embeddings,
        evaluate_supervised_classifier,
        train_supervised_classifier,
    )
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
            "PrefixLabelDataset",
            "RNNEncoder",
            "RecurrentEncoder",
            "SupervisedRNNClassifier",
            "add_embedding_norm",
            "binary_pos_weight",
            "build_bag_of_events_features",
            "build_user_embeddings",
            "build_user_embeddings_with_pooling",
            "build_activity_bucket",
            "build_base_meta",
            "build_baseline_features",
            "build_full_history_baseline_features",
            "build_meta_for_prefix",
            "build_meta_for_full_history",
            "build_supervised_embeddings",
            "build_random_embeddings",
            "choose_device",
            "cluster_profile",
            "embedding_columns",
            "embedding_matrix",
            "evaluate_baseline",
            "evaluate_supervised_classifier",
            "evaluate_next_event_model",
            "evaluate_xgboost_feature_set",
            "fit_kmeans",
            "pca_projection",
            "pool_hidden_states",
            "ranking_metrics_from_logits",
            "seed_everything",
            "shuffle_user_prefixes",
            "shuffle_user_sequences",
            "train_supervised_classifier",
            "train_model",
        ]
    )
