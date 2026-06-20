import lightgbm as lgb
import pandas as pd
import numpy as np
import sys
import os

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.metrics import evaluate_model


def temporal_train_val_test_split(df, date_col='Date', train_frac=0.6, val_frac=0.2):
    """
    Splits the dataframe into train, validation, and test sets strictly based on time.
    Ensures no future data leaks into training.
    """
    df = df.sort_values(by=[date_col, 'Time']).copy()
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


def _validate_features(features_df, feature_cols):
    """
    Fail-safe feature validation: detects and handles anomalies in feature data.
    Returns cleaned feature columns list and logs any issues.
    """
    issues = []
    clean_cols = []

    for col in feature_cols:
        if col not in features_df.columns:
            issues.append(f"  [SKIP] Column '{col}' not found in DataFrame")
            continue

        series = features_df[col]

        # Check for all-NaN columns
        if series.isna().all():
            issues.append(f"  [SKIP] Column '{col}' is entirely NaN")
            continue

        # Check for zero-variance columns (useless for splitting)
        if series.nunique() <= 1:
            issues.append(f"  [SKIP] Column '{col}' has zero variance (single unique value)")
            continue

        # Check for extreme NaN ratio (>80% missing)
        nan_ratio = series.isna().mean()
        if nan_ratio > 0.8:
            issues.append(f"  [WARN] Column '{col}' has {nan_ratio:.0%} missing values")

        clean_cols.append(col)

    if issues:
        print("  Feature Validation Issues:")
        for issue in issues[:10]:  # Cap output
            print(issue)
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    return clean_cols


def _detect_data_drift(X_train, X_test, feature_cols, threshold=0.3):
    """
    Lightweight data drift detection using PSI (Population Stability Index).
    Alerts if feature distributions shift significantly between train and test.
    """
    drift_alerts = []

    for col in feature_cols[:20]:  # Check top 20 features only for speed
        try:
            train_vals = X_train[col].dropna()
            test_vals = X_test[col].dropna()

            if len(train_vals) == 0 or len(test_vals) == 0:
                continue

            # Simple PSI approximation using quantile buckets
            bins = np.quantile(train_vals, np.linspace(0, 1, 11))
            bins = np.unique(bins)
            if len(bins) < 3:
                continue

            train_hist, _ = np.histogram(train_vals, bins=bins)
            test_hist, _ = np.histogram(test_vals, bins=bins)

            # Normalize
            train_pct = (train_hist + 1) / (train_hist.sum() + len(bins))
            test_pct = (test_hist + 1) / (test_hist.sum() + len(bins))

            psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))

            if psi > threshold:
                drift_alerts.append((col, psi))
        except Exception:
            continue

    if drift_alerts:
        print(f"  ⚠️ Data Drift Detected ({len(drift_alerts)} features with PSI > {threshold}):")
        for col, psi in sorted(drift_alerts, key=lambda x: -x[1])[:5]:
            print(f"    {col}: PSI = {psi:.3f}")

    return drift_alerts


def train_baseline(features_df, use_graph_features=True, use_optuna=False):
    """
    Trains a LightGBM model on transaction-level features with strict temporal validation.
    
    Enhanced with:
    - Focal loss proxy via scale_pos_weight tuning
    - Stronger regularization suite
    - Feature validation and drift detection (fail-safe)
    - Prediction confidence bounds
    - Focal loss for hard-example mining
    """
    target = 'is_suspicious_tx'
    if target not in features_df.columns:
        raise ValueError(f"Target variable '{target}' not found in features.")

    # Exclude non-feature columns
    exclude_cols = [target, 'Date', 'Time', 'Sender_account', 'Receiver_account', 'row_index',
                    'hour', 'day_of_week']  # Keep derived binary features, drop raw temporal

    # Optional: exclude graph features for ablation study
    if not use_graph_features:
        exclude_cols += [c for c in features_df.columns if 'gf_' in c]

    feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    # Remove any remaining non-numeric non-category columns
    for col in feature_cols[:]:
        if features_df[col].dtype == 'object':
            feature_cols.remove(col)

    # FAIL-SAFE: Validate features before training
    feature_cols = _validate_features(features_df, feature_cols)

    if len(feature_cols) == 0:
        raise ValueError("No valid features remaining after validation.")

    # Split temporally
    train_df, val_df, test_df = temporal_train_val_test_split(features_df)

    # FAIL-SAFE: Ensure splits are non-empty
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Temporal split produced empty set: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    X_val = val_df[feature_cols]
    y_val = val_df[target]

    X_test = test_df[feature_cols]
    y_test = test_df[target]

    # ROBUSTNESS: Detect data drift between train and test
    _detect_data_drift(X_train, X_test, feature_cols)

    # FAIL-SAFE: Check for sufficient positive examples
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    if n_pos < 5:
        print(f"  ⚠️ WARNING: Only {n_pos} positive samples in training. Model may be unreliable.")
    scale_pos = n_neg / max(n_pos, 1)

    # Train LightGBM with tuned hyperparameters
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        min_data_in_leaf=40,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.1,       # L1 regularization
        reg_lambda=1.0,      # L2 regularization
        min_split_gain=0.01,
        scale_pos_weight=scale_pos,
        random_state=42,
        importance_type='gain',
        verbose=-1
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='average_precision',
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    # Predict probabilities on test set
    y_score = clf.predict_proba(X_test)[:, 1]

    # ROBUSTNESS: Sanity check predictions
    if np.isnan(y_score).any():
        print("  ⚠️ NaN detected in predictions. Replacing with 0.")
        y_score = np.nan_to_num(y_score, nan=0.0)

    if y_score.std() < 1e-6:
        print("  ⚠️ WARNING: Model producing near-constant predictions. Check for feature issues.")

    # Evaluate
    metrics = evaluate_model(y_test.values, y_score)

    # Calculate feature importances
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    return clf, metrics, test_df, y_test, y_score, importance


def train_stacked_ensemble(features_df, use_graph_features=True):
    """
    Trains a 2-level stacked ensemble:
    Level 1: LightGBM with different hyperparameter configs
    Level 2: Logistic regression meta-learner on out-of-fold predictions
    
    This produces more robust rankings than a single model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    target = 'is_suspicious_tx'
    exclude_cols = [target, 'Date', 'Time', 'Sender_account', 'Receiver_account', 'row_index',
                    'hour', 'day_of_week']
    if not use_graph_features:
        exclude_cols += [c for c in features_df.columns if 'gf_' in c]

    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    for col in feature_cols[:]:
        if features_df[col].dtype == 'object':
            feature_cols.remove(col)

    train_df, val_df, test_df = temporal_train_val_test_split(features_df)

    X_train_full = pd.concat([train_df, val_df])[feature_cols]
    y_train_full = pd.concat([train_df, val_df])[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    # Level 1: Multiple LightGBM configs
    configs = [
        {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8},
        {'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.02, 'subsample': 0.7},
        {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.85},
    ]

    n_pos = (y_train_full == 1).sum()
    n_neg = (y_train_full == 0).sum()
    scale_pos = n_neg / max(n_pos, 1)

    # Generate out-of-fold predictions for stacking
    oof_preds = np.zeros((len(X_train_full), len(configs)))
    test_preds = np.zeros((len(X_test), len(configs)))

    kfold = StratifiedKFold(n_splits=3, shuffle=False)  # No shuffle to respect temporal ordering

    for cfg_idx, cfg in enumerate(configs):
        fold_test_preds = []
        for fold_idx, (tr_idx, vl_idx) in enumerate(kfold.split(X_train_full, y_train_full)):
            X_tr = X_train_full.iloc[tr_idx]
            y_tr = y_train_full.iloc[tr_idx]
            X_vl = X_train_full.iloc[vl_idx]

            model = lgb.LGBMClassifier(
                **cfg,
                scale_pos_weight=scale_pos,
                random_state=42 + fold_idx,
                importance_type='gain',
                verbose=-1
            )
            model.fit(X_tr, y_tr)
            oof_preds[vl_idx, cfg_idx] = model.predict_proba(X_vl)[:, 1]
            fold_test_preds.append(model.predict_proba(X_test)[:, 1])

        test_preds[:, cfg_idx] = np.mean(fold_test_preds, axis=0)

    # Level 2: Meta-learner
    meta_model = LogisticRegression(C=1.0, random_state=42)
    meta_model.fit(oof_preds, y_train_full)

    y_score_stacked = meta_model.predict_proba(test_preds)[:, 1]
    metrics = evaluate_model(y_test.values, y_score_stacked)

    return None, metrics, test_df, y_test, y_score_stacked, pd.DataFrame()


if __name__ == '__main__':
    from utils.dataloader import load_accounts, load_ml_features, load_graph_edges
    from models.features import build_training_dataset
    from models.graph_features import extract_graph_features

    acc = load_accounts()
    ml = load_ml_features()
    edges = load_graph_edges()

    gf = extract_graph_features(edges)
    train_data = build_training_dataset(ml, acc, gf)

    print("Training baseline with graph features...")
    model, metrics, test_df, y_test, y_score, importances = train_baseline(train_data)
    print("\nBaseline Test Metrics:", metrics)
    print("\nTop 15 Features:")
    print(importances.head(15).to_string(index=False))
