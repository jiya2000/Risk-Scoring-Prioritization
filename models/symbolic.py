"""
Symbolic AML Rule Engine — Maps flagged account behaviors to recognized
financial crime typologies using the Experta expert system library.

Typologies:
1. Smurfing (Structuring) - Many small transactions under reporting threshold
2. Fan-In - Multiple sources converging to single sink account
3. Fan-Out - Single source distributing to many recipients
4. Rapid Movement - Funds pass through account almost immediately
5. Cross-Border Burst - High-value funds moving across jurisdictions
6. Circular Flow - Account participates in closed transaction loops
7. Layering - Deep chain of intermediaries
8. Dormant Activation - Previously inactive account with sudden activity
9. Round-Trip - Funds return to origin through intermediaries
"""

import collections
import collections.abc
# Compatibility shim for Python 3.10+ with experta
collections.Mapping = collections.abc.Mapping

from experta import *
import pandas as pd
import numpy as np


class AccountFact(Fact):
    """Information about an account's recent activity."""
    pass


class AMLRuleEngine(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.risk_adjustments = []
        self.explanations = []
        self.typologies = []

    def reset_results(self):
        self.risk_adjustments = []
        self.explanations = []
        self.typologies = []

    # ─────────────────────────────────────────────────────────────────
    # RULE 1: Smurfing (Structuring)
    # Many incoming transfers, each under the reporting threshold
    # ─────────────────────────────────────────────────────────────────
    @Rule(AccountFact(count_in=P(lambda x: x >= 10), total_in=P(lambda x: x < 100000)))
    def check_smurfing_strict(self):
        self.risk_adjustments.append(20)
        self.explanations.append("High volume of sub-threshold incoming transfers (structuring pattern)")
        self.typologies.append("Smurfing")

    @Rule(AccountFact(count_in=P(lambda x: x >= 5), avg_in_amount=P(lambda x: x < 50000)))
    def check_smurfing_moderate(self):
        self.risk_adjustments.append(12)
        self.explanations.append("Multiple incoming transfers with individually low amounts")
        self.typologies.append("Smurfing (Moderate)")

    # ─────────────────────────────────────────────────────────────────
    # RULE 2: Fan-In
    # Many unique senders funneling into one account
    # ─────────────────────────────────────────────────────────────────
    @Rule(AccountFact(unique_senders=P(lambda x: x >= 8), count_out=P(lambda y: y <= 3)))
    def check_fan_in(self):
        self.risk_adjustments.append(18)
        self.explanations.append("Multiple sources converging to a single account (Fan-In collection)")
        self.typologies.append("Fan-In")

    # ─────────────────────────────────────────────────────────────────
    # RULE 3: Fan-Out
    # One account distributing to many recipients
    # ─────────────────────────────────────────────────────────────────
    @Rule(AccountFact(unique_receivers=P(lambda x: x >= 8), count_in=P(lambda y: y <= 3)))
    def check_fan_out(self):
        self.risk_adjustments.append(18)
        self.explanations.append("Funds dispersing from single account to many destinations (Fan-Out distribution)")
        self.typologies.append("Fan-Out")

    # ─────────────────────────────────────────────────────────────────
    # RULE 4: Rapid Movement
    # Almost all incoming funds immediately transferred out
    # ─────────────────────────────────────────────────────────────────
    @Rule(AccountFact(flow_ratio=P(lambda x: x >= 0.90), count_in=P(lambda y: y >= 3)))
    def check_rapid_movement(self):
        self.risk_adjustments.append(15)
        self.explanations.append("Almost all incoming funds are immediately transferred out (pass-through)")
        self.typologies.append("Rapid Movement")

    # ─────────────────────────────────────────────────────────────────
    # RULE 5: Cross-Border Burst
    # High proportion of funds moving cross-border
    # ─────────────────────────────────────────────────────────────────
    @Rule(AccountFact(cross_border_ratio_out=P(lambda x: x >= 0.5), total_out=P(lambda y: y > 100000)))
    def check_cross_border_burst(self):
        self.risk_adjustments.append(22)
        self.explanations.append("High proportion of high-value funds moving cross-border")
        self.typologies.append("Cross-Border Burst")

    # ─────────────────────────────────────────────────────────────────
    # RULE 6: Circular Flow
    # Account participates in a directed cycle
    # ─────────────────────────────────────────────────────────────────
    @Rule(AccountFact(is_cycle=True))
    def check_circular_flow(self):
        self.risk_adjustments.append(25)
        self.explanations.append("Account is part of a closed circular transaction flow")
        self.typologies.append("Circular Flow")

    # ─────────────────────────────────────────────────────────────────
    # RULE 7: Layering
    # Deep chain of intermediaries
    # ─────────────────────────────────────────────────────────────────
    @Rule(AccountFact(layering_depth=P(lambda x: x >= 3)))
    def check_layering(self):
        self.risk_adjustments.append(22)
        self.explanations.append("Account is part of a deep layering chain (3+ hops)")
        self.typologies.append("Layering")

    # ─────────────────────────────────────────────────────────────────
    # RULE 8: Dormant Activation
    # Sudden burst after period of inactivity
    # ─────────────────────────────────────────────────────────────────
    @Rule(AccountFact(activity_ratio=P(lambda x: x >= 5.0)))
    def check_dormant_activation(self):
        self.risk_adjustments.append(16)
        self.explanations.append("Sudden activation after period of dormancy (activity spike 5× baseline)")
        self.typologies.append("Dormant Activation")

    # ─────────────────────────────────────────────────────────────────
    # RULE 9: High-Value Single Transfer
    # Single transaction with unusually high amount
    # ─────────────────────────────────────────────────────────────────
    @Rule(AccountFact(max_single_tx=P(lambda x: x >= 1000000), count_in=P(lambda y: y <= 2)))
    def check_high_value_single(self):
        self.risk_adjustments.append(14)
        self.explanations.append("Single high-value transfer (>1M NPR) with minimal other activity")
        self.typologies.append("High-Value Transfer")

    # ─────────────────────────────────────────────────────────────────
    # RULE 10: Reciprocal Flow (potential Round-Trip)
    # High bidirectional flow between same counterparties
    # ─────────────────────────────────────────────────────────────────
    @Rule(AccountFact(reciprocity=P(lambda x: x >= 0.7), count_in=P(lambda y: y >= 3)))
    def check_round_trip(self):
        self.risk_adjustments.append(18)
        self.explanations.append("High bidirectional flow indicating potential round-trip laundering")
        self.typologies.append("Round-Trip")


def build_account_facts(account_txs_df, account_id, graph_features=None):
    """
    Constructs an AccountFact dict from raw transaction data for a given account.
    This bridges the gap between ml_features columns and the Experta fact schema.
    
    Args:
        account_txs_df: DataFrame of transactions involving this account
        account_id: The account being evaluated
        graph_features: Optional dict/Series of precomputed graph features
    """
    if len(account_txs_df) == 0:
        return {}

    # Incoming transactions (account is receiver)
    incoming = account_txs_df[account_txs_df.get('Receiver_account', pd.Series()) == account_id]
    outgoing = account_txs_df[account_txs_df.get('Sender_account', pd.Series()) == account_id]

    count_in = len(incoming)
    count_out = len(outgoing)
    total_in = incoming['amount_local_npr'].sum() if 'amount_local_npr' in incoming.columns and count_in > 0 else 0
    total_out = outgoing['amount_local_npr'].sum() if 'amount_local_npr' in outgoing.columns and count_out > 0 else 0

    facts = {
        'count_in': count_in,
        'count_out': count_out,
        'total_in': float(total_in),
        'total_out': float(total_out),
        'unique_senders': int(incoming['Sender_account'].nunique()) if count_in > 0 else 0,
        'unique_receivers': int(outgoing['Receiver_account'].nunique()) if count_out > 0 else 0,
        'avg_in_amount': float(total_in / max(count_in, 1)),
        'flow_ratio': float(total_out / max(total_in, 1)) if total_in > 0 else 0,
        'max_single_tx': float(account_txs_df['amount_local_npr'].max()) if 'amount_local_npr' in account_txs_df.columns else 0,
    }

    # Cross-border ratio
    if 'cross_border_flag' in account_txs_df.columns and count_out > 0:
        cb_out = outgoing['cross_border_flag'].sum()
        facts['cross_border_ratio_out'] = float(cb_out / max(count_out, 1))
    else:
        facts['cross_border_ratio_out'] = 0.0

    # Activity ratio (recent vs historical) — proxy for dormant activation
    if 'tx_count_10' in account_txs_df.columns and 'tx_count_30' in account_txs_df.columns:
        recent = account_txs_df['tx_count_10'].mean()
        historical = account_txs_df['tx_count_30'].mean()
        if historical > 0:
            facts['activity_ratio'] = float(recent / historical)
        else:
            facts['activity_ratio'] = float(recent) if recent > 0 else 0

    # Graph-based facts
    facts['is_cycle'] = False
    facts['layering_depth'] = 0
    facts['reciprocity'] = 0.0

    if graph_features is not None:
        if 'gf_in_cycle' in graph_features:
            facts['is_cycle'] = bool(graph_features['gf_in_cycle'] > 0)
        if 'gf_reciprocity' in graph_features:
            facts['reciprocity'] = float(graph_features['gf_reciprocity'])

    return facts


def evaluate_rules(account_row):
    """
    Evaluates AML rules for a single account row (dict or Series).
    Returns total adjustment, combined typologies, and explanations.
    
    Accepts either:
    - A pre-built fact dict (from build_account_facts)
    - A raw account row (backwards compatible)
    """
    engine = AMLRuleEngine()
    engine.reset()
    engine.reset_results()

    # Convert row to dict, handling NaNs
    if isinstance(account_row, pd.Series):
        fact_data = {k: v for k, v in account_row.to_dict().items() if pd.notnull(v)}
    elif isinstance(account_row, dict):
        fact_data = {k: v for k, v in account_row.items() if v is not None and (not isinstance(v, float) or not np.isnan(v))}
    else:
        fact_data = dict(account_row)

    # Defaults for required facts
    if 'is_cycle' not in fact_data:
        fact_data['is_cycle'] = False
    if 'layering_depth' not in fact_data:
        fact_data['layering_depth'] = 0

    # Filter to only valid fact keys (avoid passing DataFrame column names that aren't facts)
    valid_keys = {
        'count_in', 'count_out', 'total_in', 'total_out',
        'unique_senders', 'unique_receivers', 'avg_in_amount',
        'flow_ratio', 'cross_border_ratio_out', 'is_cycle',
        'layering_depth', 'activity_ratio', 'max_single_tx', 'reciprocity'
    }
    filtered_facts = {k: v for k, v in fact_data.items() if k in valid_keys}

    try:
        engine.declare(AccountFact(**filtered_facts))
        engine.run()
    except Exception:
        # Graceful fallback if engine fails
        pass

    total_adj = sum(engine.risk_adjustments)
    typologies = ", ".join(sorted(set(engine.typologies))) if engine.typologies else "None"
    explanations = " | ".join(engine.explanations) if engine.explanations else "No anomalous rules triggered."

    return total_adj, typologies, explanations


if __name__ == '__main__':
    # Test cases
    print("=" * 60)
    print("  SYMBOLIC RULE ENGINE TESTS")
    print("=" * 60)

    tests = [
        ("Smurfing", {'count_in': 16, 'total_in': 45000, 'avg_in_amount': 2800, 'unique_senders': 3, 'count_out': 1}),
        ("Fan-In", {'unique_senders': 12, 'count_out': 1, 'count_in': 15, 'total_in': 500000}),
        ("Fan-Out", {'unique_receivers': 10, 'count_in': 2, 'count_out': 12, 'total_out': 300000}),
        ("Rapid Movement", {'flow_ratio': 0.98, 'count_in': 8, 'total_in': 200000}),
        ("Cross-Border", {'cross_border_ratio_out': 0.9, 'total_out': 500000, 'count_out': 5}),
        ("Circular", {'is_cycle': True, 'count_in': 5}),
        ("Round-Trip", {'reciprocity': 0.8, 'count_in': 5}),
        ("Clean Account", {'count_in': 2, 'total_in': 80000, 'count_out': 1}),
    ]

    for name, facts in tests:
        adj, typ, exp = evaluate_rules(facts)
        status = "⚠️" if adj > 0 else "✅"
        print(f"\n  {status} {name}: +{adj} | {typ}")
        if adj > 0:
            print(f"     {exp}")
