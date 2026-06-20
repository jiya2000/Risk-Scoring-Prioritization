# Evaluation Methodology & Data Leakage Prevention Report

## Executive Summary
This document details the rigorous evaluation methodology employed to ensure our AML Risk Scoring system produces genuine, production-realistic metrics with zero temporal data leakage.

---

## 1. Temporal Data Leakage Mitigation

### Problem
Financial crime models commonly suffer from look-ahead bias, where future information influences past predictions.

### Implementation
We enforce a strict **chronological temporal split** with no shuffling:
- **Training Set (60%):** Earliest transactions
- **Validation Set (20%):** Middle period (used for early stopping)
- **Test Set (20%):** Most recent transactions (never seen during training)

### Graph Feature Temporal Cutoff
Graph features (PageRank, centrality, cycles) are computed **only from training-period edges**:
- The cutoff date is determined dynamically from the training split boundary
- Edges from the validation/test period are excluded from graph construction
- Graph features are then mapped statically to validation/test transactions
- This prevents topological leakage (future transactions altering historical metrics)

---

## 2. Preventing ID-Memorization (Graph Regularization)

### Problem
LightGBM can split on exact float values (e.g., PageRank = 0.001243), effectively using graph metrics as unique account identifiers — memorizing the training set rather than learning generalizable patterns.

### Solution: Quantile Binning
All continuous graph features are discretized into deciles (10 bins) using rank-first strategy:
- Forces the model to learn structural **thresholds** ("account is in top 10% of centrality")
- Eliminates the ability to memorize individual account IDs via exact float matching
- Ratio features (degree_ratio, flow_ratio, reciprocity) use 5 bins for finer granularity

### Regularization Parameters
```python
LGBMClassifier(
    max_depth=6,
    min_data_in_leaf=40,
    reg_alpha=0.1,        # L1
    reg_lambda=1.0,       # L2
    min_split_gain=0.01,
    subsample=0.75,
    colsample_bytree=0.75
)
```

---

## 3. Two-Track Evaluation (Synthetic Artifact Isolation)

### Motivation
Synthetic datasets often contain unintentional labeling artifacts (e.g., account age is a perfect predictor because the data generator used it to assign labels). To demonstrate **genuine value of our Graph Intelligence**, we run two evaluation tracks:

### Track 1: Full Feature Set (Production)
Uses all available features including potential synthetic correlations.
This is the production model that would be deployed.

### Track 2: Artifact-Reduced (Challenge Setting)
Removes features suspected of synthetic correlation (account_age_days, tx_count_30).
Under these strict conditions, the Graph Intelligence layer's contribution can be isolated and measured independently.

### Expected Results

| Model Configuration | Precision@50 | Purpose |
|:---|:---:|:---|
| Track 1: Full + Graph + Rules | High | Production deployment model |
| Track 2: Baseline Only | Low | Establishes genuine task difficulty |
| Track 2: + Graph Intelligence | Higher | Proves graph features add real value |
| Track 2: + Graph + Symbolic Rules | Highest | Shows full system synergy |

---

## 4. Score Fusion Strategy

### Architecture
```
LightGBM Score ──┐
                  ├── Sigmoid-scaled Rule Bumps ──→ Fused Score ──→ Account Aggregation
Symbolic Rules ──┘
```

### Rule Adjustment Scaling
Symbolic rule adjustments (0-100 scale) are converted to probability bumps via sigmoid squashing:
```
bump = 0.25 * (1 - exp(-adjustment / 50))
```
This provides diminishing returns for high adjustments and prevents score saturation.

### Account-Level Aggregation
Transaction scores are aggregated to accounts using a composite strategy:
- 30% Exponential time-decay weighted average (half-life = 3 days)
- 25% Top-5 transaction mean
- 20% Maximum single transaction score
- 15% High-risk transaction ratio (fraction > 0.5)
- 10% Burst score (max inter-transaction risk increase)

---

## 5. Symbolic Layer Role

The symbolic rule engine serves as the **Explanation Engine** (not a ranking engine):
- Maps ML-flagged behaviors to recognized legal typologies
- Provides human-interpretable reasons for analyst investigation
- Moderate score uplift for rule-triggering transactions ensures typology-matching accounts don't fall below the investigation threshold

### Supported Typologies
1. Smurfing (Structuring)
2. Fan-In (Collection)
3. Fan-Out (Distribution)
4. Rapid Movement (Pass-through)
5. Cross-Border Burst
6. Circular Flow (via graph cycle detection)
7. Layering (deep chain)
8. Dormant Activation
9. High-Value Transfer
10. Round-Trip (reciprocal flows)

---

## 6. Graph Feature Set (16 Features)

| Feature | Description | AML Relevance |
|:---|:---|:---|
| gf_in_degree | Incoming connection count | Identifies collectors |
| gf_out_degree | Outgoing connection count | Identifies distributors |
| gf_weighted_in_degree | Incoming weighted by amount | High-value collection |
| gf_weighted_out_degree | Outgoing weighted by amount | High-value distribution |
| gf_pagerank | Weighted PageRank centrality | Core network nodes |
| gf_clustering_coefficient | Local clustering | Tight fraud clusters |
| gf_betweenness | Betweenness centrality | Bridge/intermediary nodes |
| gf_hub_score | HITS hub score | Key distributors |
| gf_authority_score | HITS authority score | Key collectors |
| gf_degree_ratio | in/(in+out) ratio | Sink vs source behavior |
| gf_flow_ratio | weighted_in/(weighted_in + weighted_out) | Flow direction |
| gf_unique_in_neighbors | Distinct senders | Counterparty diversity |
| gf_unique_out_neighbors | Distinct receivers | Distribution breadth |
| gf_reciprocity | Bilateral flow fraction | Round-trip detection |
| gf_in_cycle | SCC membership (>1 node) | Closed loops |

---

## 7. Metrics Used

- **Precision@K** (K=10, 50, 100): Primary metric — "of the top K, how many are truly suspicious?"
- **Recall@K**: "What fraction of all fraud appears in the top K?"
- **NDCG@K**: Position-sensitive ranking quality
- **Lift@K**: Improvement vs random baseline
- **AUC-PR**: Overall ranking quality across all thresholds
- **Bootstrap 95% CI**: Statistical confidence on key metrics

---

## 8. Fairness & Bias Considerations

- No demographic features (gender, ethnicity) are used in modeling
- KYC risk grades are institution-assigned and used as aggregate rates (not individual flags)
- The model is purely behavioral — flags are based on transaction patterns, network structure, and temporal anomalies
- Priority bands are quantile-based (relative ranking), not absolute thresholds that could systematically disadvantage account segments
