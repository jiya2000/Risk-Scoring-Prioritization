"""
Explainability Module — SHAP-based interpretability for AML risk scores.

Provides:
- Global feature importance via SHAP TreeExplainer
- Per-account risk factor attribution
- Human-readable risk driver narratives
- Integration with symbolic rules for combined explanations
"""

import shap
import pandas as pd
import numpy as np


def get_shap_explanation(model, X):
    """
    Computes SHAP values for a given model and dataset.
    
    Returns:
        explainer: SHAP TreeExplainer instance
        shap_values: SHAP values for positive class (suspicious)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For LightGBM binary classification, shap_values is a list [class_0, class_1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    return explainer, shap_values_pos


def get_top_risk_factors(explainer, shap_values, X, account_index, top_k=5):
    """
    Extracts the top K features contributing positively to the risk score
    for a specific account/transaction.
    
    Returns list of dicts with 'feature', 'value', 'shap_value', and 'direction'.
    """
    feature_names = X.columns
    account_shap = shap_values[account_index]

    contributions = []
    for i, name in enumerate(feature_names):
        contributions.append({
            'feature': name,
            'value': X.iloc[account_index, i],
            'shap_value': account_shap[i],
            'direction': 'risk_increasing' if account_shap[i] > 0 else 'risk_decreasing'
        })

    # Sort by absolute SHAP value (most impactful first)
    contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

    # Filter only positive (risk-increasing) contributions
    positive_contribs = [c for c in contributions if c['shap_value'] > 0]

    return positive_contribs[:top_k]


def feature_name_to_readable(feature_name):
    """
    Converts internal feature names to human-readable descriptions.
    """
    mapping = {
        'velocity_sum_10tx': 'Transaction velocity (10-tx window)',
        'amount_local_npr': 'Transaction amount (NPR)',
        'cross_border_flag': 'Cross-border transaction',
        'tx_count_10': 'Transaction count (10-day)',
        'tx_count_30': 'Transaction count (30-day)',
        'sender_gf_pagerank': 'Sender network centrality (PageRank)',
        'receiver_gf_pagerank': 'Receiver network centrality (PageRank)',
        'sender_gf_out_degree': 'Sender outgoing connections',
        'sender_gf_in_degree': 'Sender incoming connections',
        'receiver_gf_in_degree': 'Receiver incoming connections',
        'sender_gf_betweenness': 'Sender bridge centrality',
        'receiver_gf_betweenness': 'Receiver bridge centrality',
        'sender_gf_hub_score': 'Sender hub score (distributor)',
        'receiver_gf_authority_score': 'Receiver authority score (collector)',
        'sender_gf_in_cycle': 'Sender in transaction cycle',
        'sender_institution_risk': 'Sender institution risk rating',
        'receiver_institution_risk': 'Receiver institution risk rating',
        'is_off_hours': 'Off-hours transaction',
        'is_weekend': 'Weekend transaction',
        'pair_total_txs': 'Pair transaction frequency',
        'amount_zscore_sender': 'Amount deviation from sender norm',
        'near_threshold_100k': 'Near 100K threshold (structuring)',
        'near_threshold_500k': 'Near 500K threshold (structuring)',
    }
    return mapping.get(feature_name, feature_name.replace('_', ' ').title())


def generate_account_explanation(model, X, account_index, symbolic_explanations=None, top_k=5):
    """
    Generates a comprehensive human-readable explanation combining:
    1. SHAP-based top risk drivers
    2. Symbolic rule explanations
    
    Returns a structured explanation string suitable for analyst review.
    """
    explainer, shap_values = get_shap_explanation(model, X)
    top_factors = get_top_risk_factors(explainer, shap_values, X, account_index, top_k)

    explanation_parts = []

    # ML-based explanation
    if top_factors:
        factor_descriptions = []
        for f in top_factors:
            readable = feature_name_to_readable(f['feature'])
            val = f['value']
            if isinstance(val, float):
                factor_descriptions.append(f"{readable} = {val:.2f}")
            else:
                factor_descriptions.append(f"{readable} = {val}")
        
        explanation_parts.append(f"ML Risk Drivers: {'; '.join(factor_descriptions)}")

    # Symbolic explanation
    if symbolic_explanations and symbolic_explanations != "No anomalous rules triggered.":
        explanation_parts.append(f"Rule Triggers: {symbolic_explanations}")

    if not explanation_parts:
        return "Low risk profile. No significant risk drivers identified."

    return " || ".join(explanation_parts)


def get_global_feature_importance(model, X, top_k=20):
    """
    Computes global SHAP-based feature importance (mean absolute SHAP value).
    More reliable than LightGBM's built-in gain-based importance.
    """
    explainer, shap_values = get_shap_explanation(model, X)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': mean_abs_shap,
        'readable_name': [feature_name_to_readable(c) for c in X.columns]
    }).sort_values('mean_abs_shap', ascending=False)

    return importance_df.head(top_k)


if __name__ == '__main__':
    # Mock test
    import lightgbm as lgb
    
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=['velocity_sum_10tx', 'cross_border_flag', 'sender_gf_pagerank', 'amount_local_npr', 'tx_count_10'])
    y = np.random.randint(2, size=100)
    model = lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y)

    exp = generate_account_explanation(
        model, X, 0,
        "High number of incoming transfers under threshold within 24h"
    )
    print("Explanation:", exp)

    print("\nGlobal Importance:")
    print(get_global_feature_importance(model, X))
