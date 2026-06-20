"""
Evaluation Metrics Module — Implements ranking-focused metrics for AML risk scoring.

Primary metrics:
- Precision@K: Of the top K flagged items, how many are truly suspicious?
- Recall@K: Of all suspicious items, what fraction appears in the top K?
- AUC-PR: Area under Precision-Recall curve (ranking quality)
- NDCG@K: Normalized Discounted Cumulative Gain (position-sensitive ranking)
- Lift@K: How much better is the model vs random sampling?
"""

from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np


def precision_at_k(y_true, y_score, k):
    """Calculates Precision at top K."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    df = pd.DataFrame({'true': y_true, 'score': y_score})
    df = df.sort_values(by='score', ascending=False)
    top_k = df.head(k)
    if len(top_k) == 0:
        return 0.0
    return top_k['true'].sum() / k


def recall_at_k(y_true, y_score, k):
    """Calculates Recall at top K."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    df = pd.DataFrame({'true': y_true, 'score': y_score})
    df = df.sort_values(by='score', ascending=False)
    top_k = df.head(k)
    total_positives = df['true'].sum()
    if total_positives == 0:
        return 0.0
    return top_k['true'].sum() / total_positives


def ndcg_at_k(y_true, y_score, k):
    """
    Normalized Discounted Cumulative Gain at K.
    Penalizes relevant items ranked lower in the list.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Sort by predicted score
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order][:k]

    # DCG
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(y_true_sorted / discounts)

    # Ideal DCG (all positives at the top)
    ideal_order = np.argsort(-y_true)
    y_true_ideal = y_true[ideal_order][:k]
    idcg = np.sum(y_true_ideal / discounts)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def lift_at_k(y_true, y_score, k):
    """
    Lift at K: ratio of model's precision@K to the random baseline precision.
    Lift > 1 means the model is better than random.
    """
    p_at_k = precision_at_k(y_true, y_score, k)
    random_precision = np.mean(y_true)
    if random_precision == 0:
        return float('inf') if p_at_k > 0 else 1.0
    return p_at_k / random_precision


def evaluate_model(y_true, y_score):
    """
    Comprehensive evaluation using Track 4 specific metrics.
    Returns a dict of all key metrics for reporting.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    metrics = {
        'Precision@10': precision_at_k(y_true, y_score, 10),
        'Precision@50': precision_at_k(y_true, y_score, 50),
        'Precision@100': precision_at_k(y_true, y_score, 100),
        'Recall@50': recall_at_k(y_true, y_score, 50),
        'Recall@100': recall_at_k(y_true, y_score, 100),
        'NDCG@50': ndcg_at_k(y_true, y_score, 50),
        'NDCG@100': ndcg_at_k(y_true, y_score, 100),
        'Lift@50': lift_at_k(y_true, y_score, 50),
        'AUC-PR': average_precision_score(y_true, y_score),
    }
    return metrics


def brier_score(y_true, y_score):
    """
    Brier Score — measures calibration quality of probability predictions.
    Lower is better. Perfect calibration = 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    return np.mean((y_score - y_true) ** 2)


def bootstrap_metric(y_true, y_score, metric_fn, k=50, n_bootstrap=500, ci=0.95):
    """
    Computes bootstrap confidence interval for any metric function.
    Returns (mean, lower_bound, upper_bound).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    results = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            val = metric_fn(y_true[idx], y_score[idx], k)
        except Exception:
            continue
        results.append(val)

    results = np.array(results)
    alpha = (1 - ci) / 2
    return float(np.mean(results)), float(np.percentile(results, alpha * 100)), float(np.percentile(results, (1 - alpha) * 100))


def score_distribution_stats(y_score):
    """
    Returns distribution statistics of predicted scores for sanity checking.
    Useful for detecting model collapse or calibration issues.
    """
    y_score = np.asarray(y_score)
    return {
        'mean': float(np.mean(y_score)),
        'std': float(np.std(y_score)),
        'min': float(np.min(y_score)),
        'max': float(np.max(y_score)),
        'median': float(np.median(y_score)),
        'pct_above_05': float((y_score > 0.5).mean()),
        'pct_above_09': float((y_score > 0.9).mean()),
    }


def print_metrics_report(metrics, title="Model Evaluation"):
    """Pretty-prints a metrics dictionary."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test metrics
    y_true = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.4, 0.95, 0.85, 0.1, 0.7, 0.2, 0.3, 0.5])

    metrics = evaluate_model(y_true, y_score)
    print_metrics_report(metrics, "Test Metrics")
