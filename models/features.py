import pandas as pd
import numpy as np


def compute_institution_branch_risk(accounts_df):
    """
    Computes risk rates for institutions and branches based on the KYC risk_grade.
    """
    risk_map = {'RISK-LOW': 1, 'RISK-MED': 2, 'RISK-HIGH': 3}
    accounts_df = accounts_df.copy()
    accounts_df['numeric_risk'] = accounts_df['risk_grade'].map(risk_map).fillna(1)

    inst_risk = accounts_df.groupby('institution')['numeric_risk'].mean().rename('institution_risk_rate')
    branch_risk = accounts_df.groupby('branch')['numeric_risk'].mean().rename('branch_risk_rate')

    acc_enriched = accounts_df.join(inst_risk, on='institution').join(branch_risk, on='branch')
    return acc_enriched


def compute_temporal_features(ml_features):
    """
    Engineers temporal features from Date/Time columns.
    These capture time-of-day and day-of-week patterns common in money laundering.
    """
    df = ml_features.copy()

    # Parse time components
    if 'Time' in df.columns:
        # Extract hour from time string (format: HH:MM:SS or similar)
        df['hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
        if df['hour'].isna().all():
            # Try alternate parsing
            df['hour'] = df['Time'].astype(str).str[:2].astype(float, errors='ignore')
        df['hour'] = df['hour'].fillna(12).astype(int)

        # Binary: is the transaction outside business hours (before 9am or after 5pm)?
        df['is_off_hours'] = ((df['hour'] < 9) | (df['hour'] >= 17)).astype(int)

        # Binary: is the transaction during late night (11pm - 5am)?
        df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] < 5)).astype(int)

    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'], errors='coerce')
        df['day_of_week'] = dates.dt.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    return df


def compute_interaction_features(ml_features):
    """
    Computes sender-receiver pairwise interaction features.
    Identifies repeated corridors and unusual pair activity.
    """
    df = ml_features.copy()

    # Count how many times this sender-receiver pair transacted (within the dataset)
    pair_counts = df.groupby(['Sender_account', 'Receiver_account']).cumcount()
    df['pair_tx_sequence'] = pair_counts  # 0-indexed sequence number for this pair

    # Total transactions per pair (as a feature)
    pair_totals = df.groupby(['Sender_account', 'Receiver_account'])['amount_local_npr'].transform('count')
    df['pair_total_txs'] = pair_totals

    # Mean amount for this pair (deviation from pair norm)
    pair_mean_amt = df.groupby(['Sender_account', 'Receiver_account'])['amount_local_npr'].transform('mean')
    df['pair_amount_deviation'] = (df['amount_local_npr'] - pair_mean_amt).abs()

    # Is this pair active only once? (one-shot corridors are suspicious)
    df['is_one_shot_pair'] = (df['pair_total_txs'] == 1).astype(int)

    return df


def compute_velocity_features(ml_features):
    """
    Computes additional velocity and burst-detection features.
    """
    df = ml_features.copy()

    # Ratio of recent velocity to longer-term velocity
    if 'velocity_sum_10tx' in df.columns and 'tx_count_30' in df.columns:
        df['velocity_burst_ratio'] = df['velocity_sum_10tx'] / (df['tx_count_30'] * df['amount_local_npr'].clip(lower=1))
        df['velocity_burst_ratio'] = df['velocity_burst_ratio'].clip(upper=100).fillna(0)

    # Amount relative to account mean (z-score proxy)
    sender_mean = df.groupby('Sender_account')['amount_local_npr'].transform('mean')
    sender_std = df.groupby('Sender_account')['amount_local_npr'].transform('std').fillna(1).clip(lower=1)
    df['amount_zscore_sender'] = (df['amount_local_npr'] - sender_mean) / sender_std

    # Is amount just below common reporting thresholds? (Structuring detection)
    # Common thresholds: 100K NPR, 500K NPR, 1M NPR
    for threshold in [100000, 500000, 1000000]:
        col_name = f'near_threshold_{threshold // 1000}k'
        df[col_name] = ((df['amount_local_npr'] >= threshold * 0.8) &
                        (df['amount_local_npr'] < threshold)).astype(int)

    return df


def build_training_dataset(ml_features, accounts, graph_features):
    """
    Merges KYC, Graph, Temporal, Interaction, and Velocity features into
    the transaction-level dataset for model training.
    """
    # 1. Compute KYC risk rates
    accounts = compute_institution_branch_risk(accounts)

    # 2. Engineer additional features
    ml_features = compute_temporal_features(ml_features)
    ml_features = compute_interaction_features(ml_features)
    ml_features = compute_velocity_features(ml_features)

    # 3. Merge Sender Account KYC Features
    acc_cols = ['account_id', 'institution_risk_rate', 'branch_risk_rate', 'risk_grade']
    sender_features = accounts[acc_cols].rename(columns={
        'institution_risk_rate': 'sender_institution_risk',
        'branch_risk_rate': 'sender_branch_risk',
        'risk_grade': 'sender_kyc_risk_grade'
    })
    ml_features = ml_features.merge(sender_features, left_on='Sender_account', right_on='account_id', how='left')
    if 'account_id' in ml_features.columns:
        ml_features.drop('account_id', axis=1, inplace=True)

    # 4. Merge Receiver Account KYC Features
    receiver_features = accounts[acc_cols].rename(columns={
        'institution_risk_rate': 'receiver_institution_risk',
        'branch_risk_rate': 'receiver_branch_risk',
        'risk_grade': 'receiver_kyc_risk_grade'
    })
    ml_features = ml_features.merge(receiver_features, left_on='Receiver_account', right_on='account_id', how='left')
    if 'account_id' in ml_features.columns:
        ml_features.drop('account_id', axis=1, inplace=True)

    # 5. Merge Sender Graph Features
    if len(graph_features) > 0 and 'account_id' in graph_features.columns:
        gf_cols = graph_features.columns.tolist()
        sender_gf = graph_features.rename(columns={c: f"sender_{c}" for c in gf_cols if c != 'account_id'})
        ml_features = ml_features.merge(sender_gf, left_on='Sender_account', right_on='account_id', how='left')
        if 'account_id' in ml_features.columns:
            ml_features.drop('account_id', axis=1, inplace=True)

        # 6. Merge Receiver Graph Features
        receiver_gf = graph_features.rename(columns={c: f"receiver_{c}" for c in gf_cols if c != 'account_id'})
        ml_features = ml_features.merge(receiver_gf, left_on='Receiver_account', right_on='account_id', how='left')
        if 'account_id' in ml_features.columns:
            ml_features.drop('account_id', axis=1, inplace=True)

    # Convert categorical strings to categorical dtype for LightGBM
    for col in ['sender_kyc_risk_grade', 'receiver_kyc_risk_grade']:
        if col in ml_features.columns:
            ml_features[col] = ml_features[col].astype('category')

    # Fill NaN graph features with 0 (accounts not in graph)
    gf_fill_cols = [c for c in ml_features.columns if 'gf_' in c]
    ml_features[gf_fill_cols] = ml_features[gf_fill_cols].fillna(0)

    return ml_features


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.dataloader import load_accounts, load_ml_features, load_graph_edges
    from models.graph_features import extract_graph_features

    acc = load_accounts()
    ml = load_ml_features()
    edges = load_graph_edges()

    gf = extract_graph_features(edges)
    train_df = build_training_dataset(ml, acc, gf)
    print("Final Training Data Shape:", train_df.shape)
    print("Columns:", train_df.columns.tolist())
    print("\nNew feature samples:")
    new_cols = ['is_off_hours', 'is_weekend', 'pair_total_txs', 'amount_zscore_sender', 'near_threshold_100k']
    available = [c for c in new_cols if c in train_df.columns]
    print(train_df[available].describe())
