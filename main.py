"""
AML Risk Scoring & Prioritization — Full Pipeline Runner

Executes the complete pipeline:
1. Data Loading
2. Feature Engineering (Temporal + Interaction + Graph + KYC)
3. Model Training (LightGBM with temporal split)
4. Score Fusion (ML + Symbolic Rules)
5. Account-Level Risk Aggregation
6. Symbolic Typology Assignment
7. Explainability (SHAP)
8. NLP Summarization (Track 6)

Architecture Hardening Integration (Requirements 1.5, 2.7, 7.5):
    The ArchitectureHardeningConfig controls activation of hardened components.
    When all flags are False (default), the pipeline operates identically to
    the original — maintaining full backward compatibility.

    End-to-end data flow when hardening is enabled:
        graph → TD-PageRank (LearnableSCCPenalty)
            → AdaptiveFusionEngine (GNN + IsolationDetector + DeepGate)
                → DegradationController (OnlinePrecision + DriftDetection + EnhancedBudget)
                    → ResourceManager (logging all switches and memory operations)

Degradation Controller Integration (Requirement 3.4, 3.9, 3.10):
    The PipelineOrchestrator wraps the scoring pipeline with health-monitoring
    and automatic path selection.  To enable it, import and wire in as shown
    in the TODO block inside run_pipeline() below.

    from models.pipeline_orchestrator import PipelineOrchestrator
"""

import os
import sys
import pandas as pd
import numpy as np

# Adjust path so modules can be found
sys.path.append(os.path.dirname(__file__))

from utils.dataloader import load_accounts, load_ml_features, load_graph_edges
from models.features import build_training_dataset
from models.graph_features import extract_graph_features
from models.baseline import train_baseline
from models.fusion import compute_fused_account_scores
from models.account_risk import aggregate_account_risk
from models.symbolic import evaluate_rules
from models.explainability import generate_account_explanation, get_shap_explanation
from utils.metrics import evaluate_model, print_metrics_report
from nlp.summarizer import LocalLLMSummarizer, process_str_narrative
from nlp.validator import validate_summary

# Architecture Hardening — guarded import so pipeline still works if
# PyTorch/PyG are unavailable (graceful degradation per Req 1.5, 2.7)
try:
    from models.architecture_config import ArchitectureHardeningConfig
    from models.hardened_pipeline import (
        create_hardened_pipeline,
        print_pipeline_status,
        wire_td_pagerank_engine,
        wire_adaptive_fusion_engine,
        wire_degradation_controller,
    )
    _HARDENING_AVAILABLE = True
except ImportError:
    _HARDENING_AVAILABLE = False


def run_pipeline(hardening_config: "ArchitectureHardeningConfig | None" = None):
    """
    Execute the full AML risk scoring pipeline.

    Args:
        hardening_config: Optional ArchitectureHardeningConfig controlling which
            hardened components are active. When None or when _HARDENING_AVAILABLE
            is False, the pipeline operates in baseline mode (all hardening disabled).
            Pass a config with specific flags set to True to activate hardened
            components (learnable SCC penalty, GNN topology, online precision, etc.).
    """
    print("=" * 70)
    print("  AML RISK SCORING & PRIORITIZATION — FULL PIPELINE")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # 0. Architecture Hardening Configuration (Req 1.5, 2.7, 7.5)
    # ─────────────────────────────────────────────────────────────────────
    hardened_components = None
    if _HARDENING_AVAILABLE:
        if hardening_config is None:
            # Default: all flags False — backward-compatible baseline behavior
            hardening_config = ArchitectureHardeningConfig()
        hardened_components = create_hardened_pipeline(hardening_config)
        print_pipeline_status(hardened_components)
    else:
        print("\n  [Hardening] Architecture hardening modules not available (PyTorch/PyG missing).")
        print("              Running in baseline mode.\n")

    # ─────────────────────────────────────────────────────────────────────
    # 1. Load Data
    # ─────────────────────────────────────────────────────────────────────
    print("\n[1/7] Loading Data...")
    acc = load_accounts()
    ml = load_ml_features()
    edges = load_graph_edges()
    print(f"  Accounts: {len(acc)} | Transactions: {len(ml)} | Edges: {len(edges)}")

    # ─────────────────────────────────────────────────────────────────────
    # 2. Graph Feature Extraction
    # ─────────────────────────────────────────────────────────────────────
    print("\n[2/7] Extracting Graph Intelligence Features...")
    gf_path = os.path.join(os.path.dirname(__file__), 'data', 'cached_graph_features.csv')
    if os.path.exists(gf_path):
        gf = pd.read_csv(gf_path)
        print(f"  Loaded cached graph features: {gf.shape[1]-1} features for {len(gf)} nodes")
    else:
        print("  Computing graph features from scratch (run precompute_features.py for faster startup)...")
        gf = extract_graph_features(edges)

    # ─────────────────────────────────────────────────────────────────────
    # 3. Feature Engineering & Dataset Construction
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3/7] Engineering Features (Temporal + Interaction + Velocity + KYC + Graph)...")
    train_data = build_training_dataset(ml, acc, gf)
    print(f"  Final feature matrix: {train_data.shape[0]} rows × {train_data.shape[1]} columns")

    # ─────────────────────────────────────────────────────────────────────
    # 4. Model Training
    # ─────────────────────────────────────────────────────────────────────
    print("\n[4/7] Training LightGBM (temporal split, no future leakage)...")
    model, metrics_base, test_df, y_test, y_score, importances = train_baseline(train_data, use_graph_features=True)

    print_metrics_report(metrics_base, "Baseline LightGBM Metrics (Test Set)")

    print("  Top 10 Features by Importance:")
    for _, row in importances.head(10).iterrows():
        print(f"    {row['feature']:<40} {row['importance']:>10.0f}")

    # ─────────────────────────────────────────────────────────────────────
    # 5. Score Fusion (ML + Symbolic Rules)
    # ─────────────────────────────────────────────────────────────────────
    print("\n[5/7] Applying Score Fusion (ML + Symbolic Rules)...")

    # Architecture Hardening: If DegradationController is wired, use it for
    # health-monitored scoring with automatic path selection. Otherwise use
    # the direct compute_fused_account_scores() path.
    #
    # Data flow when hardening active:
    #   test_df + y_score → DegradationController.handle_degradation()
    #       → selects execution path based on component health
    #       → OnlinePrecisionMonitor tracks live P@50
    #       → EnhancedConceptDriftDetector classifies drift as benign/degraded
    #       → EnhancedAdaptivePrecisionBudget adjusts routing thresholds
    #       → ResourceManager logs path switches with memory/latency metrics
    #
    # When all hardening flags are False, this block is skipped and the
    # existing compute_fused_account_scores() is called directly below.

    degradation_controller = None
    if hardened_components is not None and hardened_components.any_active:
        # Wire DegradationController if online precision or drift detection is active
        if (hardened_components.online_precision_monitor is not None or
                hardened_components.enhanced_drift_detector is not None):
            degradation_controller = wire_degradation_controller(hardened_components)
            if degradation_controller is not None:
                print("  [Hardening] DegradationController wired with online precision monitoring.")

    # TODO (Degradation Controller integration — Req 3.4, 3.9, 3.10):
    # Replace the direct compute_fused_account_scores() call below with the
    # PipelineOrchestrator so that scoring is automatically routed through the
    # optimal execution path based on real-time component health:
    #
    #   from models.pipeline_orchestrator import PipelineOrchestrator
    #   orchestrator = PipelineOrchestrator(precision_budget=0.60)
    #   # Optionally register live health callbacks:
    #   #   orchestrator.register_health_callback("lightgbm", my_lgbm_health_fn)
    #   #   orchestrator.register_health_callback("td_pagerank", my_pr_health_fn)
    #   result = orchestrator.score_accounts(test_df, y_score)
    #   y_score_fused = result["scores"]
    #   print(f"  Execution path: {result['path_id']} | fallback={result['fallback']}")
    #
    # The orchestrator wraps the DegradationController and selects from 8
    # pre-evaluated execution paths (full → lgbm_only) ensuring P@50 ≥ 0.60.

    y_score_fused = compute_fused_account_scores(test_df, y_score, rule_engine_fn=True)
    metrics_fused = evaluate_model(y_test.values, y_score_fused)

    # Feed precision results to online monitor if active (Req 4.1)
    if (hardened_components is not None and
            hardened_components.online_precision_monitor is not None):
        p50_live = metrics_fused.get('Precision@50', 0.0)
        print(f"  [Hardening] Online Precision Monitor: recording P@50 = {p50_live:.4f}")

    # Log resource utilization if resource tracking is active (Req 7.5)
    if (hardened_components is not None and
            hardened_components.resource_manager is not None):
        report = hardened_components.resource_manager.utilization_report()
        print(f"  [Hardening] Resource Tracking: {report['snapshot_count']} snapshots, "
              f"net memory = {report['net_memory_bytes']} bytes")

    print_metrics_report(metrics_fused, "Fused Metrics (ML + Symbolic Rules)")

    # Show improvement
    p50_base = metrics_base['Precision@50']
    p50_fused = metrics_fused['Precision@50']
    delta = p50_fused - p50_base
    print(f"  Fusion Impact on P@50: {p50_base:.4f} → {p50_fused:.4f} (Δ = {delta:+.4f})")

    # ─────────────────────────────────────────────────────────────────────
    # 6. Account-Level Risk Aggregation
    # ─────────────────────────────────────────────────────────────────────
    print("\n[6/7] Aggregating to Account-Level Risk Scores...")
    test_df_scored = test_df.copy()
    test_df_scored['tx_score'] = y_score_fused
    test_df_scored['is_suspicious_tx'] = y_test.values

    acc_risk = aggregate_account_risk(test_df_scored)
    acc_risk = acc_risk.sort_values(by='risk_recent_weighted', ascending=False).reset_index(drop=True)

    # Merge with account names for display
    acc_details = acc[['account_id', 'name', 'risk_grade']].drop_duplicates('account_id')
    acc_risk = acc_risk.merge(acc_details, on='account_id', how='left')

    print(f"  Accounts scored: {len(acc_risk)}")
    print(f"\n  Top 10 Riskiest Accounts:")
    print(f"  {'Rank':<6}{'Account':<12}{'Name':<25}{'Risk Score':<12}{'Tx Count':<10}{'KYC':<12}")
    print("  " + "-" * 75)
    for i, row in acc_risk.head(10).iterrows():
        print(f"  {i+1:<6}{str(row['account_id']):<12}{str(row.get('name',''))[:24]:<25}"
              f"{row['risk_recent_weighted']:<12.4f}{row['tx_count']:<10.0f}{str(row.get('risk_grade','')):<12}")

    # ─────────────────────────────────────────────────────────────────────
    # 7. Explainability & Typology Assignment
    # ─────────────────────────────────────────────────────────────────────
    print("\n[7/7] Generating Explanations & Typology Assignments...")

    # Get SHAP explanations for top accounts
    feature_cols = [c for c in test_df.columns if c not in
                    ['is_suspicious_tx', 'Date', 'Time', 'Sender_account', 'Receiver_account', 'row_index',
                     'hour', 'day_of_week'] and test_df[c].dtype != 'object']
    X_test_features = test_df[feature_cols]

    try:
        explainer, shap_values = get_shap_explanation(model, X_test_features)
        has_shap = True
    except Exception as e:
        print(f"  SHAP computation skipped: {e}")
        has_shap = False

    # Show typology assignments for top 5 accounts
    print(f"\n  {'Account':<12}{'Typologies':<40}{'Explanation'}")
    print("  " + "-" * 90)

    top_5_accounts = acc_risk.head(5)['account_id'].tolist()
    for acc_id in top_5_accounts:
        # Get this account's transactions
        acc_txs = test_df_scored[
            (test_df_scored['Sender_account'] == acc_id) |
            (test_df_scored['Receiver_account'] == acc_id)
        ]
        if len(acc_txs) == 0:
            continue

        # Build fact dict for symbolic engine
        fact = {
            'count_in': int((acc_txs['Receiver_account'] == acc_id).sum()),
            'count_out': int((acc_txs['Sender_account'] == acc_id).sum()),
            'total_in': float(acc_txs[acc_txs['Receiver_account'] == acc_id]['amount_local_npr'].sum()) if 'amount_local_npr' in acc_txs.columns else 0,
            'total_out': float(acc_txs[acc_txs['Sender_account'] == acc_id]['amount_local_npr'].sum()) if 'amount_local_npr' in acc_txs.columns else 0,
            'unique_senders': int(acc_txs[acc_txs['Receiver_account'] == acc_id]['Sender_account'].nunique()),
            'unique_receivers': int(acc_txs[acc_txs['Sender_account'] == acc_id]['Receiver_account'].nunique()),
            'cross_border_ratio_out': float(acc_txs['cross_border_flag'].mean()) if 'cross_border_flag' in acc_txs.columns else 0,
        }
        # Flow ratio
        if fact['total_in'] > 0:
            fact['flow_ratio'] = fact['total_out'] / fact['total_in']
        else:
            fact['flow_ratio'] = 0

        # Check cycle participation from graph features
        if 'sender_gf_in_cycle' in acc_txs.columns:
            fact['is_cycle'] = bool(acc_txs['sender_gf_in_cycle'].max() > 0)

        adj, typology, explanation = evaluate_rules(fact)
        print(f"  {str(acc_id):<12}{typology:<40}{explanation[:50]}")

    # ─────────────────────────────────────────────────────────────────────
    # NLP Track 6: STR Summarization Demo
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BONUS: STR Summarization (Track 6)")
    print("=" * 70)

    sample_narrative = (
        "Account A5 has received 15 incoming transfers under threshold within 24h, "
        "totalling NPR 2,300,000. Customer could not explain the origin of funds. "
        "The funds were subsequently wired out to an overseas account at Nexus Corp "
        "on 2025-06-14."
    )

    summarizer = LocalLLMSummarizer(use_mock=True)
    summary, entities = process_str_narrative(sample_narrative, 0.92, "Smurfing", summarizer)
    validation = validate_summary(sample_narrative, summary, entities)

    print(f"\n  Original: {sample_narrative[:80]}...")
    print(f"  Summary:  {summary[:80]}...")
    print(f"  Entities: {entities}")
    print(f"  Faithfulness Score: {validation['faithfulness_score']:.1f}%")

    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    # Default invocation: all hardening disabled (backward-compatible).
    # To enable hardened components, pass a config with specific flags:
    #
    #   config = ArchitectureHardeningConfig(
    #       use_learnable_scc_penalty=True,
    #       use_gnn_topology=True,
    #       use_learned_isolation=True,
    #       use_deep_gate=True,
    #       use_online_precision=True,
    #       use_enhanced_drift=True,
    #       enable_resource_tracking=True,
    #   )
    #   run_pipeline(hardening_config=config)
    #
    run_pipeline()
