import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataloader import load_accounts, load_ml_features, load_graph_edges
from models.features import build_training_dataset
from models.graph_features import extract_graph_features
from models.baseline import train_baseline

def run_audit():
    ml = load_ml_features()
    acc = load_accounts()
    edges = load_graph_edges()
    
    print("="*50)
    print("1. & 5. Temporal Split & Class Distribution")
    print("="*50)
    # Replicate the split
    df = ml.sort_values(by=['Date', 'Time']).copy()
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    print(f"Train: {train['Date'].min()} to {train['Date'].max()} | Suspicious: {train['is_suspicious_tx'].sum()}/{len(train)} ({train['is_suspicious_tx'].mean():.4f})")
    print(f"Val:   {val['Date'].min()} to {val['Date'].max()} | Suspicious: {val['is_suspicious_tx'].sum()}/{len(val)} ({val['is_suspicious_tx'].mean():.4f})")
    print(f"Test:  {test['Date'].min()} to {test['Date'].max()} | Suspicious: {test['is_suspicious_tx'].sum()}/{len(test)} ({test['is_suspicious_tx'].mean():.4f})")
    
    print("\n" + "="*50)
    print("6. Unique Accounts")
    print("="*50)
    train_accs = set(train['Sender_account']).union(set(train['Receiver_account']))
    test_accs = set(test['Sender_account']).union(set(test['Receiver_account']))
    overlap = train_accs.intersection(test_accs)
    print(f"Train Unique Accounts: {len(train_accs)}")
    print(f"Test Unique Accounts: {len(test_accs)}")
    print(f"Overlap (Accounts in both): {len(overlap)}")
    
    print("\n" + "="*50)
    print("4. Precision@50 using ONLY Baseline vs ONLY Graph vs BOTH")
    print("="*50)
    gf = extract_graph_features(edges)
    graph_df = build_training_dataset(df, acc, gf)
    
    # Baseline Only
    _, met_base, _, _, _, _ = train_baseline(graph_df, use_graph_features=False)
    
    # Graph Only
    # To do this, we manually drop all features except graph and target
    target = 'is_suspicious_tx'
    gf_cols = [c for c in graph_df.columns if 'gf_' in c]
    exclude = [c for c in graph_df.columns if c not in gf_cols and c not in [target, 'Date', 'Time', 'Sender_account', 'Receiver_account']]
    
    # Temporary patch to train_baseline
    import models.baseline
    # We will just write a tiny training loop here to ensure we get the metrics
    from models.baseline import temporal_train_val_test_split
    from utils.metrics import evaluate_model
    
    t_train, t_val, t_test = temporal_train_val_test_split(graph_df)
    
    def train_custom(features):
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, class_weight='balanced', random_state=42, importance_type='gain')
        clf.fit(t_train[features], t_train[target])
        y_score = clf.predict_proba(t_test[features])[:, 1]
        return evaluate_model(t_test[target], y_score), clf
        
    met_gf_only, _ = train_custom(gf_cols)
    
    _, met_both, _, _, _, imp_both = train_baseline(graph_df, use_graph_features=True)
    
    print(f"Baseline Only P@50: {met_base['Precision@50']:.4f}")
    print(f"Graph Only P@50:    {met_gf_only['Precision@50']:.4f}")
    print(f"Both P@50:          {met_both['Precision@50']:.4f}")
    
    print("\n" + "="*50)
    print("2. Top 20 LGBM Features (Both)")
    print("="*50)
    print(imp_both.head(20).to_string(index=False))

if __name__ == '__main__':
    run_audit()
