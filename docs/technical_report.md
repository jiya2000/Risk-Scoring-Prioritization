# AI/ML Intelligence Hackathon - Technical Report

## Track Selection
We have developed a combined solution addressing **Track 4 (Risk Scoring & Prioritization)** and **Track 6 (AI-Powered Analysis & Reporting)**.

## Architecture

### 1. Feature Engineering
We extracted behavioral signals such as Flow Ratio, Counterparty Churn, Inter-Transaction Time Variance, and Cross-Border transfer ratios.

### 2. Primary Risk Model (Track 4)
We utilize a LightGBM classification baseline combined with a Symbolic Rule Engine (Experta) evaluating 7 distinct AML typologies (Smurfing, Layering, Circular Flow, Fan-In, Fan-Out, Rapid Movement, Cross-Border Burst). 

### 3. Research Extension: Heterogeneous GNN
We built an experimental PyTorch Geometric Heterogeneous GNN that models accounts and KYC entities as nodes, with transactions as edges. We implemented Ego-IDs and Port-Numbering to handle multi-edges.

### 4. Explainability
SHAP values are extracted to provide the analyst with the top tabular risk drivers, which are presented alongside the triggered typology rules.

### 5. Entity-Faithful Summarization (Track 6)
We pre-process STR narratives with spaCy NER to extract amounts, dates, and accounts. A local open-source LLM summarizes the text, injecting the Track 4 risk score context. Finally, an Entity Faithfulness Validator ensures no extracted entities were lost or hallucinated.

## Evaluation
See `evaluation.ipynb` for final metrics (Precision@K, Recall@50, AUC-PR).
