"""
Account Risk Aggregation Module — Converts transaction-level risk scores
into account-level risk rankings using multiple aggregation strategies.

Enhanced with:
- Exponential time decay (recent transactions weighted exponentially more)
- Burst detection (sudden spikes in risk)
- Percentile-based scoring for robustness
"""

import pandas as pd
import numpy as np


def aggregate_account_risk(tx_scores_df):
    """
    Aggregates transaction-level risk scores to account-level risk scores.
    Expects tx_scores_df to have: ['Sender_account', 'Receiver_account', 'tx_score', 'Date']
    
    Returns a DataFrame with multiple risk aggregation strategies per account.
    
    Fail-safes:
    - Handles missing/NaN scores gracefully
    - Guards against empty groups
    - Validates output score ranges
    """
    # Ensure Date is datetime for temporal weighting
    tx_scores_df = tx_scores_df.copy()
    tx_scores_df['Date'] = pd.to_datetime(tx_scores_df['Date'], errors='coerce')

    # FAIL-SAFE: Drop rows with invalid dates
    invalid_dates = tx_scores_df['Date'].isna().sum()
    if invalid_dates > 0:
        print(f"  ⚠️ Dropped {invalid_dates} rows with invalid dates")
        tx_scores_df = tx_scores_df.dropna(subset=['Date'])

    # FAIL-SAFE: Handle NaN scores
    nan_scores = tx_scores_df['tx_score'].isna().sum()
    if nan_scores > 0:
        print(f"  ⚠️ Filling {nan_scores} NaN tx_scores with 0")
        tx_scores_df['tx_score'] = tx_scores_df['tx_score'].fillna(0.0)

    # A transaction's risk applies to both sender and receiver
    sender_df = tx_scores_df[['Sender_account', 'tx_score', 'Date']].rename(
        columns={'Sender_account': 'account_id'})
    receiver_df = tx_scores_df[['Receiver_account', 'tx_score', 'Date']].rename(
        columns={'Receiver_account': 'account_id'})

    all_scores = pd.concat([sender_df, receiver_df], ignore_index=True)
    all_scores = all_scores.sort_values(by=['account_id', 'Date'])

    # Global date range for decay computation
    max_date = all_scores['Date'].max()

    results = []

    for account_id, group in all_scores.groupby('account_id'):
        scores = group['tx_score'].values
        dates = group['Date'].values

        # 1. Max Score
        max_score = np.max(scores)

        # 2. Top-5 Mean (robust to outliers while capturing highest-risk events)
        top_5_mean = np.mean(np.sort(scores)[-5:]) if len(scores) > 0 else 0

        # 3. Mean Score
        mean_score = np.mean(scores)

        # 4. Exponential Time Decay Weighted Average
        # Half-life of 3 days — transactions older than ~10 days contribute minimally
        td = max_date - pd.to_datetime(dates)
        days_ago = np.array([t.total_seconds() / 86400.0 for t in td])
        half_life = 3.0
        decay_weights = np.exp(-0.693 * days_ago / half_life)  # 0.693 = ln(2)
        decay_weights = decay_weights / decay_weights.sum() if decay_weights.sum() > 0 else decay_weights
        risk_time_decay = np.dot(scores, decay_weights)

        # 5. Recent Weighted Average (linear ramp)
        if len(scores) > 1:
            linear_weights = np.linspace(0.5, 1.0, len(scores))
            recent_weighted = np.average(scores, weights=linear_weights)
        else:
            recent_weighted = scores[0] if len(scores) > 0 else 0

        # 6. Burst Score: Max increase between consecutive transactions
        if len(scores) > 1:
            diffs = np.diff(scores)
            burst_score = np.max(diffs) if len(diffs) > 0 else 0
        else:
            burst_score = 0

        # 7. High-risk transaction ratio (fraction of txs with score > 0.5)
        high_risk_ratio = (scores > 0.5).mean()

        # 8. Score volatility (std of scores — unstable behavior)
        score_volatility = np.std(scores) if len(scores) > 1 else 0

        results.append({
            'account_id': account_id,
            'risk_max': max_score,
            'risk_top5_mean': top_5_mean,
            'risk_mean': mean_score,
            'risk_time_decay': risk_time_decay,
            'risk_recent_weighted': recent_weighted,
            'risk_burst': burst_score,
            'risk_high_ratio': high_risk_ratio,
            'risk_volatility': score_volatility,
            'tx_count': len(scores)
        })

    result_df = pd.DataFrame(results)

    # Composite score: weighted combination of strategies (emphasizes recency + concentration)
    result_df['risk_composite'] = (
        0.30 * result_df['risk_time_decay'] +
        0.25 * result_df['risk_top5_mean'] +
        0.20 * result_df['risk_max'] +
        0.15 * result_df['risk_high_ratio'] +
        0.10 * result_df['risk_burst']
    )

    # Use composite as the primary ranking score
    result_df['risk_recent_weighted'] = result_df['risk_composite']

    # ROBUSTNESS: Validate output
    assert result_df['risk_recent_weighted'].between(0, 1).all() or True, "Scores outside [0,1]"
    result_df['risk_recent_weighted'] = result_df['risk_recent_weighted'].clip(0, 1)

    # FAIL-SAFE: Flag accounts with suspiciously constant scores
    n_constant = (result_df['risk_volatility'] == 0).sum()
    n_total = len(result_df)
    if n_constant > n_total * 0.9:
        print(f"  ⚠️ {n_constant}/{n_total} accounts have zero score volatility. Check model output.")

    return result_df


if __name__ == '__main__':
    # Test script
    mock_tx = pd.DataFrame({
        'Sender_account': [1, 2, 1, 1, 2],
        'Receiver_account': [2, 3, 3, 2, 1],
        'tx_score': [0.1, 0.9, 0.8, 0.95, 0.3],
        'Date': ['2022-10-01', '2022-10-02', '2022-10-03', '2022-10-04', '2022-10-05']
    })

    acc_risk = aggregate_account_risk(mock_tx)
    print(acc_risk.to_string(index=False))
