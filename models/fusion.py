"""
Score Fusion Module — Combines ML, Graph, and Symbolic scores into a
single calibrated risk probability.

Strategies:
1. Weighted fusion with learned or fixed weights
2. Rank-based fusion (RRF - Reciprocal Rank Fusion) for robust ranking
3. Isotonic calibration for well-calibrated final probabilities
4. Adaptive fusion via AdaptiveFusionEngine (topology-conditioned weights)
"""

from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd
import sys
import os
import logging

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def fuse_scores(tabular_scores, gnn_scores=None, rule_adjustments=None,
                tabular_weight=0.7, gnn_weight=0.3):
    """
    Fuses scores from different model sources.
    If GNN scores are missing (due to fallback), uses tabular + rules only.
    
    Args:
        tabular_scores: LightGBM predicted probabilities
        gnn_scores: Optional GNN predicted probabilities
        rule_adjustments: Integer adjustments from symbolic engine (0-100 scale)
        tabular_weight: Weight for tabular model
        gnn_weight: Weight for GNN model
    """
    fused = np.array(tabular_scores, dtype=np.float64)

    # Ensemble with GNN if available
    if gnn_scores is not None:
        fused = tabular_weight * fused + gnn_weight * np.array(gnn_scores)

    # Apply rule adjustments as probability bumps
    if rule_adjustments is not None:
        # Scale: max possible adjustment ~ 100 → maps to 0.0-1.0 bump
        # Use sigmoid-like scaling so large adjustments have diminishing returns
        adj = np.array(rule_adjustments, dtype=np.float64)
        # Sigmoid squash: maps 0→0, 50→~0.12, 100→~0.23
        rule_bump = 0.25 * (1 - np.exp(-adj / 50.0))
        fused = fused + rule_bump

    return np.clip(fused, 0.0, 1.0)


def reciprocal_rank_fusion(score_lists, k=60):
    """
    Reciprocal Rank Fusion (RRF) — a robust rank aggregation method that
    combines multiple ranked lists without needing score calibration.
    
    Used when different models produce scores on different scales.
    
    Args:
        score_lists: List of arrays, each containing scores for the same items
        k: RRF constant (default 60, standard in literature)
    
    Returns:
        fused_scores: Array of fused scores (higher = riskier)
    """
    n = len(score_lists[0])
    rrf_scores = np.zeros(n)

    for scores in score_lists:
        # Convert scores to ranks (highest score = rank 1)
        ranks = np.argsort(np.argsort(-np.array(scores))) + 1  # 1-indexed ranks
        rrf_scores += 1.0 / (k + ranks)

    return rrf_scores


def calibrate_scores(y_true, y_score_fused):
    """
    Applies Isotonic Regression to calibrate fused scores into true probabilities.
    Monotonic calibration ensures ranking is preserved while probabilities are meaningful.
    """
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_score_fused, y_true)
    calibrated_scores = iso_reg.predict(y_score_fused)

    return iso_reg, calibrated_scores


def compute_fused_account_scores(test_df, y_score, rule_engine_fn=None):
    """
    End-to-end fusion pipeline that:
    1. Takes transaction-level ML scores
    2. Applies symbolic rule adjustments per transaction
    3. Validates output score distribution (fail-safe)
    4. Returns fused scores ready for account aggregation
    
    Args:
        test_df: DataFrame with test transactions (must include feature columns)
        y_score: ML model predicted probabilities for test_df
        rule_engine_fn: Optional callable(row) -> (adjustment, typology, explanation)
    """
    fused = np.array(y_score, dtype=np.float64)

    # FAIL-SAFE: Handle NaN/Inf in input scores
    nan_count = np.isnan(fused).sum()
    if nan_count > 0:
        print(f"  ⚠️ {nan_count} NaN values in ML scores. Replacing with median.")
        median_score = np.nanmedian(fused)
        fused = np.nan_to_num(fused, nan=median_score)

    inf_count = np.isinf(fused).sum()
    if inf_count > 0:
        print(f"  ⚠️ {inf_count} Inf values in ML scores. Clipping.")
        fused = np.clip(fused, 0.0, 1.0)

    if rule_engine_fn is not None:
        # Vectorized rule application (simplified typology checks)
        adjustments = np.zeros(len(test_df))

        # Smurfing: high tx count, low amounts
        if 'tx_count_10' in test_df.columns and 'amount_local_npr' in test_df.columns:
            mask = (test_df['tx_count_10'].values >= 10) & (test_df['amount_local_npr'].values < 50000)
            adjustments[mask] += 15

        # Cross-border burst
        if 'cross_border_flag' in test_df.columns and 'amount_local_npr' in test_df.columns:
            mask = (test_df['cross_border_flag'].values == 1) & (test_df['amount_local_npr'].values > 100000)
            adjustments[mask] += 20

        # Fan-Out (high out-degree sender)
        if 'sender_gf_out_degree' in test_df.columns:
            mask = test_df['sender_gf_out_degree'].values >= 8
            adjustments[mask] += 15

        # Rapid Movement (high velocity)
        if 'velocity_sum_10tx' in test_df.columns:
            mask = test_df['velocity_sum_10tx'].values > 1000000
            adjustments[mask] += 12

        # Core network node (high pagerank)
        if 'sender_gf_pagerank' in test_df.columns:
            mask = test_df['sender_gf_pagerank'].values >= 8
            adjustments[mask] += 10

        # Cycle participation
        if 'sender_gf_in_cycle' in test_df.columns:
            mask = test_df['sender_gf_in_cycle'].values == 1
            adjustments[mask] += 18

        # Near-threshold structuring
        if 'near_threshold_100k' in test_df.columns:
            mask = test_df['near_threshold_100k'].values == 1
            adjustments[mask] += 8

        # Off-hours transactions (weak signal, small bump)
        if 'is_off_hours' in test_df.columns:
            mask = test_df['is_off_hours'].values == 1
            adjustments[mask] += 3

        fused = fuse_scores(fused, rule_adjustments=adjustments)

    # ROBUSTNESS: Post-fusion sanity check
    if fused.std() < 1e-8:
        print("  ⚠️ WARNING: Fused scores have near-zero variance. Fusion may have failed.")

    return fused


def compute_prediction_confidence(y_score, n_models=3):
    """
    Estimates prediction confidence based on score distribution.
    Scores near 0 or 1 have high confidence; scores near 0.5 have low confidence.
    
    Returns confidence array in [0, 1] where 1 = high confidence.
    """
    y_score = np.asarray(y_score)
    # Distance from decision boundary (0.5) normalized to [0, 1]
    confidence = 2.0 * np.abs(y_score - 0.5)
    return confidence


def compute_adaptive_fused_scores(test_df, y_score, G, rule_adjustments=None):
    """
    Adaptive fusion pipeline using topology-conditioned weights per transaction.

    Uses AdaptiveFusionEngine to compute per-transaction fused scores with dynamic
    weights inferred from each account's ego-network topology. Falls back to the
    existing static-weight fuse_scores() path if AdaptiveFusionEngine fails or if G
    is None.

    Args:
        test_df: DataFrame with test transactions. Must include a column that
                 identifies the sending account ('Sender_account' or 'account_id').
        y_score: Array-like of ML model predicted probabilities for test_df rows.
                 Values should be in [0.0, 1.0].
        G: nx.DiGraph — the directed transaction graph used for topology extraction.
           Pass None to trigger static-weight fallback for all transactions.
        rule_adjustments: Optional array-like of integer rule adjustments
                          (0–100 scale) per transaction. Used to derive s_rules
                          via sigmoid scaling: 0.25 * (1 - exp(-adj / 50)).
                          If None, s_rules defaults to 0.0 for all transactions.

    Returns:
        np.ndarray of fused scores in [0.0, 1.0], one per row in test_df.

    Fallback:
        If AdaptiveFusionEngine cannot be imported or raises any exception during
        processing, the function transparently falls back to fuse_scores() with the
        same tabular scores and rule_adjustments, preserving existing behaviour.
    """
    y_score = np.asarray(y_score, dtype=np.float64)

    # Determine the account ID column
    account_col = None
    for candidate in ('Sender_account', 'account_id', 'Account_ID'):
        if candidate in test_df.columns:
            account_col = candidate
            break

    # Pre-compute sigmoid-scaled rule scores (s_rules per transaction)
    if rule_adjustments is not None:
        adj = np.asarray(rule_adjustments, dtype=np.float64)
        s_rules_arr = 0.25 * (1.0 - np.exp(-adj / 50.0))
    else:
        s_rules_arr = np.zeros(len(y_score), dtype=np.float64)

    # Attempt to use AdaptiveFusionEngine
    try:
        from models.adaptive_fusion import AdaptiveFusionEngine

        engine = AdaptiveFusionEngine()
        fused = np.empty(len(y_score), dtype=np.float64)

        for i in range(len(y_score)):
            s_ml = float(y_score[i])
            s_rules = float(s_rules_arr[i])

            # s_graph: use the sender's gf_pagerank if available, else 0.0
            s_graph = 0.0
            if 'sender_gf_pagerank' in test_df.columns:
                s_graph = float(test_df.iloc[i].get('sender_gf_pagerank', 0.0))
            elif 'gf_pagerank' in test_df.columns:
                s_graph = float(test_df.iloc[i].get('gf_pagerank', 0.0))

            # Resolve account ID for topology extraction
            if account_col is not None:
                account_id = str(test_df.iloc[i][account_col])
            else:
                account_id = "__unknown__"

            result = engine.fuse(
                account_id=account_id,
                s_ml=s_ml,
                s_graph=s_graph,
                s_rules=s_rules,
                G=G,
            )
            fused[i] = result.fused_score

        logger.info(
            f"compute_adaptive_fused_scores: processed {len(fused)} transactions "
            f"via AdaptiveFusionEngine."
        )
        return np.clip(fused, 0.0, 1.0)

    except Exception as exc:
        logger.warning(
            f"AdaptiveFusionEngine failed ({exc}); falling back to static fuse_scores()."
        )
        return fuse_scores(y_score, rule_adjustments=rule_adjustments if rule_adjustments is not None else None)




if __name__ == '__main__':
    # Test
    y_true = np.array([0, 0, 1, 1, 0, 1])
    tabular = np.array([0.1, 0.4, 0.6, 0.8, 0.2, 0.9])
    gnn = np.array([0.2, 0.3, 0.7, 0.9, 0.1, 0.8])
    rules = np.array([0, 10, 0, 20, 0, 30])

    fused = fuse_scores(tabular, gnn, rules)
    print("Fused:", fused)

    iso, cal = calibrate_scores(y_true, fused)
    print("Calibrated:", cal)

    # Test RRF
    rrf = reciprocal_rank_fusion([tabular, gnn])
    print("RRF Scores:", rrf)
