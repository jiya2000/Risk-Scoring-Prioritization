"""
Feature Importance & Ablation Study

Produces a rigorous two-track evaluation:
- Track 1: Full feature set (production model)
- Track 2: Artifact-reduced evaluation (proves genuine graph value)

Reports improvements with statistical confidence via bootstrap.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataloader import load_accounts, load_ml_features
from models.features import build_training_dataset
from models.baseline import train_baseline
from models.fusion import compute_fused_account_scores
from utils.metrics import evaluate_model, print_metrics_report


def bootstrap_precision_at_k(y_true, y_score, k=50, n_bootstrap=200, ci=0.95):
    """
    Computes bootstrap confidence interval for Precision@K.
    Returns (mean, lower_bound, upper_bound).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    
    precisions = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        p = 0
        df = pd.DataFrame({'true': y_true[idx], 'score': y_score[idx]})
        df = df.sort_values('score', ascending=False)
        p = df.head(k)['true'].sum() / k
        precisions.append(p)
    
    precisions = np.array(precisions)
    alpha = (1 - ci) / 2
    lower = np.percentile(precisions, alpha * 100)
    upper = np.percentile(precisions, (1 - alpha) * 100)
    return np.mean(precisions), lower, upper


def run_evaluation():
    print("=" * 70)
    print("  COMPREHENSIVE ABLATION STUDY & FEATURE IMPORTANCE")
    print("=" * 70)
    
    print("\nLoading data and precomputed graph features...")
    acc = load_accounts()
    ml = load_ml_features()

    gf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cached_graph_features.csv')
    if not os.path.exists(gf_path):
        print("ERROR: Precomputed graph features not found.")
        print("Run: python scripts/precompute_features.py")
        return

    gf = pd.read_csv(gf_path)

    # ═══════════════════════════════════════════════════════════════════════
    # Track 1: FULL FEATURE SET (Production Model)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  TRACK 1: FULL FEATURE SET (Production Model)")
    print("─" * 70)
    full_df = build_training_dataset(ml.copy(), acc, gf)
    model_full, metrics_full, test_full, y_test_full, y_score_full, imp_full = train_baseline(full_df, use_graph_features=True)
    
    # Apply fusion
    y_fused_full = compute_fused_account_scores(test_full, y_score_full, rule_engine_fn=True)
    metrics_full_fused = evaluate_model(y_test_full.values, y_fused_full)
    
    print_metrics_report(metrics_full, "Track 1: LightGBM Only")
    print_metrics_report(metrics_full_fused, "Track 1: LightGBM + Symbolic Rules")

    # ═══════════════════════════════════════════════════════════════════════
    # Track 2: ARTIFACT-REDUCED (Challenge Setting)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  TRACK 2: ARTIFACT-REDUCED (Proves Genuine Graph Value)")
    print("─" * 70)
    
    reduced_ml = ml.copy()
    leaky_cols = ['sender_account_age_days', 'receiver_account_age_days', 'tx_count_30']
    existing_leaky = [c for c in leaky_cols if c in reduced_ml.columns]
    if existing_leaky:
        reduced_ml.drop(columns=existing_leaky, inplace=True)
        print(f"  Removed synthetic artifacts: {existing_leaky}")

    # 2a: Baseline Only (No Graph)
    base_df = build_training_dataset(reduced_ml.copy(), acc, pd.DataFrame(columns=['account_id']))
    _, metrics_base, _, y_test_base, y_score_base, _ = train_baseline(base_df, use_graph_features=False)

    # 2b: With Graph Features
    graph_df = build_training_dataset(reduced_ml.copy(), acc, gf)
    _, metrics_graph, test_graph, y_test_graph, y_score_graph, imp_graph = train_baseline(graph_df, use_graph_features=True)

    # 2c: With Graph + Symbolic Rules
    y_fused_graph = compute_fused_account_scores(test_graph, y_score_graph, rule_engine_fn=True)
    metrics_graph_rules = evaluate_model(y_test_graph.values, y_fused_graph)

    print_metrics_report(metrics_base, "Track 2: Baseline Only (No Graph, No Artifacts)")
    print_metrics_report(metrics_graph, "Track 2: + Graph Intelligence")
    print_metrics_report(metrics_graph_rules, "Track 2: + Graph + Symbolic Rules")

    # ═══════════════════════════════════════════════════════════════════════
    # IMPROVEMENT SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  ABLATION SUMMARY TABLE")
    print("═" * 70)
    print(f"  {'Model Configuration':<50} | {'P@50':<8} | {'P@100':<8} | {'AUC-PR':<8}")
    print("  " + "-" * 80)
    print(f"  {'Track 1: Full (Production)':<50} | {metrics_full_fused['Precision@50']:<8.4f} | {metrics_full_fused['Precision@100']:<8.4f} | {metrics_full_fused['AUC-PR']:<8.4f}")
    print(f"  {'Track 2: No Artifacts (Baseline)':<50} | {metrics_base['Precision@50']:<8.4f} | {metrics_base['Precision@100']:<8.4f} | {metrics_base['AUC-PR']:<8.4f}")
    print(f"  {'Track 2: No Artifacts + Graph':<50} | {metrics_graph['Precision@50']:<8.4f} | {metrics_graph['Precision@100']:<8.4f} | {metrics_graph['AUC-PR']:<8.4f}")
    print(f"  {'Track 2: No Artifacts + Graph + Rules':<50} | {metrics_graph_rules['Precision@50']:<8.4f} | {metrics_graph_rules['Precision@100']:<8.4f} | {metrics_graph_rules['AUC-PR']:<8.4f}")
    print("  " + "-" * 80)
    
    # Graph improvement
    base_p50 = metrics_base['Precision@50']
    graph_p50 = metrics_graph['Precision@50']
    if base_p50 > 0:
        improvement = ((graph_p50 - base_p50) / base_p50) * 100
        print(f"\n  📈 Graph Intelligence Improvement: +{improvement:.0f}% on P@50")
    
    # Bootstrap CI for the production model
    print("\n  Bootstrap 95% CI for Production P@50:")
    mean_p50, lower, upper = bootstrap_precision_at_k(y_test_full.values, y_fused_full, k=50)
    print(f"    Mean: {mean_p50:.4f} [{lower:.4f}, {upper:.4f}]")

    # ═══════════════════════════════════════════════════════════════════════
    # TOP FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  TOP 20 FEATURES (Track 1: Production Model)")
    print("─" * 70)
    print(imp_full.head(20).to_string(index=False))
    
    # Graph feature contribution
    graph_features = imp_full[imp_full['feature'].str.contains('gf_')]
    total_importance = imp_full['importance'].sum()
    graph_importance = graph_features['importance'].sum()
    print(f"\n  Graph features contribute {graph_importance/total_importance*100:.1f}% of total model gain")


if __name__ == '__main__':
    run_evaluation()
