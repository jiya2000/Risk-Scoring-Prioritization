import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataloader import load_accounts, load_ml_features
from models.features import build_training_dataset
from models.baseline import train_baseline
from utils.metrics import evaluate_model

def run_rule_analysis():
    print("Loading data...")
    acc = load_accounts()
    ml = load_ml_features()
    
    gf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cached_graph_features.csv')
    gf = pd.read_csv(gf_path)
    
    # We will use the Artifact-Reduced Evaluation for the base model
    reduced_ml = ml.copy()
    if 'sender_account_age_days' in reduced_ml.columns:
        reduced_ml.drop(columns=['sender_account_age_days', 'receiver_account_age_days', 'tx_count_30'], inplace=True, errors='ignore')
        
    graph_df = build_training_dataset(reduced_ml, acc, gf)
    model, metrics, test_df, y_test, y_score, _ = train_baseline(graph_df, use_graph_features=True)
    
    print("\n--- BASELINE METRICS (No Rules) ---")
    print(f"Precision@50:  {metrics['Precision@50']:.4f}")
    print(f"Precision@100: {metrics['Precision@100']:.4f}")
    print(f"Recall@50:     {metrics.get('Recall@50', 0):.4f}")
    print(f"Recall@100:    {metrics.get('Recall@100', 0):.4f}")
    
    # ---------------------------------------------------------
    # VECTORIZED SYMBOLIC RULES
    # ---------------------------------------------------------
    # Rule 1: Smurfing (High tx count, low individual amounts)
    r_smurfing = (test_df['tx_count_10'] >= 10) & (test_df['amount_local_npr'] < 50000)
    
    # Rule 2: Fan-In (Many senders, few receivers -> receiver account acts as sink)
    r_fan_in = (test_df['receiver_gf_in_degree'] >= 8) & (test_df['receiver_gf_out_degree'] <= 2)
    
    # Rule 3: Fan-Out (Few senders, many receivers -> sender acts as distributor)
    r_fan_out = (test_df['sender_gf_in_degree'] <= 2) & (test_df['sender_gf_out_degree'] >= 8)
    
    # Rule 4: Rapid Movement (High velocity over short period)
    r_rapid_move = (test_df['tx_count_10'] >= 5) & (test_df['velocity_sum_10tx'] > 500000)
    
    # Rule 5: Cross-Border Burst (Large cross border)
    r_cross_border = (test_df['cross_border_flag'] == 1) & (test_df['amount_local_npr'] > 100000)
    
    # Rule 6: High Centrality Core (Top decile pagerank)
    r_core_node = (test_df['sender_gf_pagerank'] == 9) | (test_df['receiver_gf_pagerank'] == 9)
    
    rules = {
        'Smurfing': r_smurfing,
        'Fan-In': r_fan_in,
        'Fan-Out': r_fan_out,
        'Rapid Movement': r_rapid_move,
        'Cross-Border Burst': r_cross_border,
        'Core Network Node': r_core_node
    }
    
    # The risk adjustment is added to the predicted PROBABILITY (0 to 1)
    # Since these are probability boosts, they must be small enough to act as tie-breakers or moderate bumps
    # Wait: if baseline P@50 is 0.28, the top 50 scores are very high. 
    # To significantly move a node into the top 50, the boost needs to be substantial (e.g. +0.20 to +0.40)
    rule_weights = {
        'Smurfing': 0.15,
        'Fan-In': 0.25,
        'Fan-Out': 0.20,
        'Rapid Movement': 0.15,
        'Cross-Border Burst': 0.30,
        'Core Network Node': 0.10
    }
    
    y_score_rules = y_score.copy()
    
    print("\n--- SYMBOLIC RULE TRIGGER ANALYSIS ---")
    print(f"{'Rule Name':<25} | {'Trigger Count':<15} | {'Avg Score Uplift':<20}")
    print("-" * 65)
    
    for name, mask in rules.items():
        triggers = mask.sum()
        weight = rule_weights[name]
        # Apply boost
        y_score_rules[mask] += weight
        
        # We cap at 1.0
        y_score_rules = np.clip(y_score_rules, 0, 1)
        
        # Avg score uplift calculation
        # If it triggers, it uplifts by `weight` (capped at 1.0)
        actual_uplift = np.mean(np.clip(y_score[mask] + weight, 0, 1) - y_score[mask]) if triggers > 0 else 0
        
        print(f"{name:<25} | {triggers:<15} | +{actual_uplift:.4f}")
        
    metrics_rules = evaluate_model(y_test.values, y_score_rules)
    
    print("\n--- FINAL METRICS (With Rules) ---")
    print(f"Precision@50:  {metrics_rules['Precision@50']:.4f} (was {metrics['Precision@50']:.4f})")
    print(f"Precision@100: {metrics_rules['Precision@100']:.4f} (was {metrics['Precision@100']:.4f})")
    print(f"Recall@50:     {metrics_rules.get('Recall@50', 0):.4f} (was {metrics.get('Recall@50', 0):.4f})")
    print(f"Recall@100:    {metrics_rules.get('Recall@100', 0):.4f} (was {metrics.get('Recall@100', 0):.4f})")

if __name__ == '__main__':
    run_rule_analysis()
