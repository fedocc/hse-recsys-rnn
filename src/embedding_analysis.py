from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def embedding_columns(columns: Sequence[str], prefix: str = "emb_") -> list[str]:
    return [col for col in columns if col.startswith(prefix)]


def add_embedding_norm(df: pl.DataFrame, embedding_cols: Sequence[str], norm_col: str = "embedding_norm") -> pl.DataFrame:
    return df.with_columns(
        pl.concat_list([pl.col(col).pow(2) for col in embedding_cols]).list.sum().sqrt().alias(norm_col)
    )


def embedding_matrix(
    df: pl.DataFrame,
    embedding_cols: Sequence[str],
    *,
    standardize: bool = True,
) -> np.ndarray:
    matrix = df.select(list(embedding_cols)).to_numpy()
    if not standardize:
        return matrix
    return StandardScaler().fit_transform(matrix)


def pca_projection(
    df: pl.DataFrame,
    embedding_cols: Sequence[str],
    *,
    n_components: int = 2,
    standardize: bool = True,
    seed: int = 42,
) -> tuple[pl.DataFrame, np.ndarray]:
    matrix = embedding_matrix(df, embedding_cols, standardize=standardize)
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(matrix)
    projection = df.select("appmetrica_device_id").with_columns(
        [pl.Series(f"pc_{idx + 1}", coords[:, idx]) for idx in range(n_components)]
    )
    return projection, pca.explained_variance_ratio_


def fit_kmeans(
    df: pl.DataFrame,
    embedding_cols: Sequence[str],
    *,
    n_clusters: int,
    seed: int = 42,
    n_init: int = 20,
) -> tuple[pl.DataFrame, float, float]:
    matrix = embedding_matrix(df, embedding_cols, standardize=True)
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    labels = model.fit_predict(matrix)
    score = silhouette_score(matrix, labels, sample_size=min(10_000, len(labels)), random_state=seed)
    assignments = df.select("appmetrica_device_id").with_columns(pl.Series("cluster", labels))
    return assignments, float(model.inertia_), float(score)


def cluster_profile(
    assignments: pl.DataFrame,
    metadata: pl.DataFrame,
    *,
    cluster_col: str = "cluster",
) -> pl.DataFrame:
    return (
        assignments.join(metadata, on="appmetrica_device_id", how="left")
        .group_by(cluster_col)
        .agg(
            [
                pl.len().alias("num_users"),
                pl.col("sequence_len").median().alias("median_sequence_len"),
                pl.col("sequence_len").mean().alias("mean_sequence_len"),
                pl.col("unique_event_count").mean().alias("mean_unique_events"),
                pl.col("repeat_ratio").mean().alias("mean_repeat_ratio"),
                pl.col("prefix_active_days").mean().alias("mean_prefix_active_days"),
                pl.col("retention_7d").mean().alias("retention_7d_rate"),
                pl.col("retention_14d").mean().alias("retention_14d_rate"),
            ]
        )
        .sort("num_users", descending=True)
    )


def build_activity_bucket(df: pl.DataFrame, source_col: str = "sequence_len") -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(source_col) < 25)
        .then(pl.lit("short"))
        .when(pl.col(source_col) < 100)
        .then(pl.lit("medium"))
        .when(pl.col(source_col) < 500)
        .then(pl.lit("long"))
        .otherwise(pl.lit("very_long"))
        .alias("activity_bucket")
    )
