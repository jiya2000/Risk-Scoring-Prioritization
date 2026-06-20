"""
Precompute Graph Features with Temporal Leakage Prevention

This script:
1. Determines the temporal cutoff from the training split
2. Filters edges to only use training-period transactions
3. Extracts comprehensive graph features (16 features)
4. Applies quantile binning to prevent ID-memorization by LightGBM
5. Caches the result to data/cached_graph_features.csv
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataloader import load_graph_edges, load_ml_features
from models.graph_features import extract_graph_features
from models.baseline import temporal_train_val_test_split


def safe_qcut(series, q=10):
    """
    Applies quantile binning with rank-first strategy to handle duplicates.
    Falls back to fewer bins if data has too few unique values.
    """
    try:
        return pd.qcut(series.rank(method='first'), q=q, labels=False)
    except ValueError:
        # If there are fewer unique values than requested bins
        n_unique = series.nunique()
        if n_unique <= 1:
            return pd.Series(0, index=series.index)
        return pd.qcut(series.rank(method='first'), q=min(q, n_unique), labels=False)


def precompute_graph_features():
    print("=" * 60)
    print("  GRAPH FEATURE PRECOMPUTATION (Leakage-Safe)")
    print("=" * 60)

    print("\nLoading data...")
    ml = load_ml_features()
    edges = load_graph_edges()
    print(f"  ML Features: {len(ml)} rows | Graph Edges: {len(edges)} rows")

    # 1. Determine temporal split cutoff from ml_features
    print("\nDetermining temporal cutoff to prevent leakage...")
    train_df, _, _ = temporal_train_val_test_split(ml)

    cutoff_date = train_df['Date'].max()
    cutoff_time = train_df[train_df['Date'] == cutoff_date]['Time'].max()
    print(f"  Graph Cutoff: Edges on or before {cutoff_date} {cutoff_time}")

    # 2. Filter edges to prevent temporal leakage
    edges['Date'] = edges['Date'].astype(str)
    edges['Time'] = edges['Time'].astype(str)

    mask = (edges['Date'] < cutoff_date) | (
        (edges['Date'] == cutoff_date) & (edges['Time'] <= cutoff_time)
    )
    train_edges = edges[mask].copy()
    print(f"  Original Edges: {len(edges)} | Training Edges (pre-cutoff): {len(train_edges)}")

    # 3. Extract Graph Features using ONLY Training Edges
    print("\nExtracting graph features...")
    gf = extract_graph_features(train_edges)

    # 4. Graph Regularization (Quantile Binning)
    print("\nApplying Graph Regularization (Decile Binning)...")
    print("  This prevents LightGBM from memorizing exact float IDs.")

    gf = gf.fillna(0)

    # Columns to bin into deciles (continuous features)
    bin_cols = [
        'gf_in_degree', 'gf_out_degree',
        'gf_weighted_in_degree', 'gf_weighted_out_degree',
        'gf_pagerank', 'gf_betweenness',
        'gf_hub_score', 'gf_authority_score',
        'gf_unique_in_neighbors', 'gf_unique_out_neighbors',
    ]

    for col in bin_cols:
        if col in gf.columns:
            gf[col] = safe_qcut(gf[col], q=10)

    # Ratio/binary features: keep as-is or bin into 5 buckets
    ratio_cols = ['gf_clustering_coefficient', 'gf_degree_ratio', 'gf_flow_ratio', 'gf_reciprocity']
    for col in ratio_cols:
        if col in gf.columns:
            gf[col] = safe_qcut(gf[col], q=5)

    # Binary features stay binary
    # gf_in_cycle is already 0/1

    # 5. Save to cache
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cached_graph_features.csv')
    gf.to_csv(output_path, index=False)

    print(f"\n  Saved: {output_path}")
    print(f"  Shape: {gf.shape[0]} nodes × {gf.shape[1]} columns")
    print(f"  Features: {[c for c in gf.columns if c != 'account_id']}")
    print("\n  Done! Dashboard and pipeline will use cached features.")


if __name__ == '__main__':
    precompute_graph_features()
