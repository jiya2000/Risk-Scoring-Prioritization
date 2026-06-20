"""
AML Intelligence Dashboard — Streamlit Analyst Interface

Features:
- Risk Prioritization Queue with dynamic bands
- Account Investigation with peer comparison
- Typology Threat Landscape
- Network Visualization (ego-graph)
- SHAP Explainability Waterfall
- Activity Momentum & Burst Detection
- One-Click Investigation Report Export
- STR Summarization (Track 6)
"""

import streamlit as st
import pandas as pd
import numpy as np

# Adjust path so modules can be found
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.symbolic import evaluate_rules
from models.fusion import compute_fused_account_scores
from nlp.summarizer import LocalLLMSummarizer, process_str_narrative
from nlp.validator import validate_summary

# Setup page
st.set_page_config(page_title="AML Intelligence Dashboard", layout="wide", page_icon="🔍")
st.title("🔍 AML Risk Scoring & Prioritization Platform")
st.caption("AI/ML Intelligence Hackathon — Financial Crime Detection Dashboard")


@st.cache_data
def get_real_data():
    from utils.dataloader import load_accounts, load_ml_features
    from models.features import build_training_dataset
    from models.baseline import train_baseline
    from models.account_risk import aggregate_account_risk
    import pandas as pd
    import os

    acc = load_accounts()
    ml = load_ml_features()

    gf_path = os.path.join(os.path.dirname(__file__), 'data', 'cached_graph_features.csv')
    if os.path.exists(gf_path):
        gf = pd.read_csv(gf_path)
    else:
        from utils.dataloader import load_graph_edges
        from models.graph_features import extract_graph_features
        edges = load_graph_edges()
        gf = extract_graph_features(edges)

    train_data = build_training_dataset(ml, acc, gf)

    model, metrics, test_df, y_test, y_score, importances = train_baseline(train_data, use_graph_features=True)

    # Apply score fusion with symbolic rules
    y_score_fused = compute_fused_account_scores(test_df, y_score, rule_engine_fn=True)

    # Score the test dataset
    test_df = test_df.copy()
    test_df['tx_score'] = y_score_fused
    test_df['is_suspicious_tx'] = y_test.values

    # Aggregate to account-level risk
    acc_risk = aggregate_account_risk(test_df)

    # Merge with account details for display
    acc_details = acc[['account_id', 'account_number', 'name', 'risk_grade', 'institution']].copy()
    acc_risk = acc_risk.merge(acc_details, on='account_id', how='left')
    acc_risk = acc_risk.sort_values(by='risk_recent_weighted', ascending=False)

    random_rate = len(test_df[test_df['is_suspicious_tx'] == 1]) / len(test_df)

    return acc_risk, test_df, metrics, random_rate, model, importances


acc_risk_df, tx_df, metrics, random_sampling_rate, trained_model, feature_importances = get_real_data()


# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC PRIORITY BANDS
# ═══════════════════════════════════════════════════════════════════════════
def assign_dynamic_priority(df):
    q99 = df['risk_recent_weighted'].quantile(0.99)
    q95 = df['risk_recent_weighted'].quantile(0.95)
    q80 = df['risk_recent_weighted'].quantile(0.80)

    def get_band(score):
        if score >= q99:
            return "🔴 Critical"
        elif score >= q95:
            return "🟠 High"
        elif score >= q80:
            return "🟡 Medium"
        else:
            return "🟢 Low"

    df['priority'] = df['risk_recent_weighted'].apply(get_band)
    return df


acc_risk_df = assign_dynamic_priority(acc_risk_df)

# Global Peer Averages
global_vel_avg = tx_df['velocity_sum_10tx'].mean()
global_cb_avg = tx_df['cross_border_flag'].mean()
global_tx_count_avg = tx_df['tx_count_10'].mean()


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📊 Analyst Overview", "🔬 Deep Investigation", "🧠 Advanced Features (NLP)"])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1: ANALYST OVERVIEW
# ─────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Operational Efficiency: Analyst Review Simulation")

    cap_accounts = 50
    flagged_df = acc_risk_df.head(cap_accounts)

    suspicious_accs = set(
        tx_df[tx_df['is_suspicious_tx'] == 1]['Sender_account'].unique().tolist() +
        tx_df[tx_df['is_suspicious_tx'] == 1]['Receiver_account'].unique().tolist()
    )

    true_suspicious_found = flagged_df['account_id'].isin(suspicious_accs).sum()
    precision_50 = (true_suspicious_found / cap_accounts) * 100
    random_expected = random_sampling_rate * cap_accounts
    improvement = ((true_suspicious_found - random_expected) / max(random_expected, 1)) * 100
    total_txs = len(tx_df)
    time_saved = ((total_txs - cap_accounts) / total_txs) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{total_txs:,}")
    c2.metric("Review Queue", f"{cap_accounts} Accounts", delta=f"-{time_saved:.1f}% Volume", delta_color="normal")
    c3.metric("Suspicious Found", f"{true_suspicious_found}")
    c4.metric("Precision@50", f"{precision_50:.1f}%")
    c5.metric("Lift vs Random", f"{improvement:.0f}×")

    st.markdown("---")

    # MAIN AREA
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Risk Prioritization Queue")

        # Typology Threat Landscape
        top_50_accs = flagged_df['account_id'].tolist()
        top_50_txs = tx_df[
            (tx_df['Sender_account'].isin(top_50_accs)) |
            (tx_df['Receiver_account'].isin(top_50_accs))
        ]

        fan_out_count = 0
        rapid_mov_count = 0
        cb_burst_count = 0
        smurfing_count = 0

        for acc in top_50_accs:
            acc_txs_subset = top_50_txs[
                (top_50_txs['Sender_account'] == acc) | (top_50_txs['Receiver_account'] == acc)
            ]
            if len(acc_txs_subset) == 0:
                continue
            mean_v = acc_txs_subset['velocity_sum_10tx'].mean()
            cb_count = acc_txs_subset['cross_border_flag'].sum()
            out_deg = acc_txs_subset['sender_gf_out_degree'].max() if 'sender_gf_out_degree' in acc_txs_subset.columns else 0
            tx_count = acc_txs_subset['tx_count_10'].mean()
            avg_amt = acc_txs_subset['amount_local_npr'].mean() if 'amount_local_npr' in acc_txs_subset.columns else 0

            if out_deg >= 8:
                fan_out_count += 1
            if cb_count > 0 and mean_v > 500000:
                cb_burst_count += 1
            if mean_v > 1000000:
                rapid_mov_count += 1
            if tx_count > 10 and avg_amt < 50000:
                smurfing_count += 1

        total_top_50 = len(top_50_accs)

        with st.expander("🌍 Typology Threat Landscape (Top 50)", expanded=True):
            st.markdown("Threat Distribution in Critical Queue:")
            st.progress(min(fan_out_count / max(total_top_50, 1), 1.0),
                        text=f"{fan_out_count}/{total_top_50} Fan-Out Activity")
            st.progress(min(rapid_mov_count / max(total_top_50, 1), 1.0),
                        text=f"{rapid_mov_count}/{total_top_50} Rapid Movement")
            st.progress(min(cb_burst_count / max(total_top_50, 1), 1.0),
                        text=f"{cb_burst_count}/{total_top_50} Cross-Border Bursts")
            st.progress(min(smurfing_count / max(total_top_50, 1), 1.0),
                        text=f"{smurfing_count}/{total_top_50} Smurfing")

        # Model Performance Summary
        with st.expander("📈 Model Performance", expanded=False):
            st.markdown(f"- **AUC-PR:** {metrics['AUC-PR']:.4f}")
            st.markdown(f"- **P@10:** {metrics['Precision@10']:.2%}")
            st.markdown(f"- **P@50:** {metrics['Precision@50']:.2%}")
            st.markdown(f"- **P@100:** {metrics['Precision@100']:.2%}")
            if 'NDCG@50' in metrics:
                st.markdown(f"- **NDCG@50:** {metrics['NDCG@50']:.4f}")

        display_df = acc_risk_df[['priority', 'account_id', 'name', 'risk_recent_weighted']].copy()
        display_df.rename(columns={'risk_recent_weighted': 'Risk Score'}, inplace=True)
        st.dataframe(display_df.head(50), hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Quick Account Investigation")
        selected_account = st.selectbox("Select Account ID:", acc_risk_df['account_id'].head(50))

        acc_data = acc_risk_df[acc_risk_df['account_id'] == selected_account].iloc[0]
        acc_txs = tx_df[
            (tx_df['Sender_account'] == selected_account) |
            (tx_df['Receiver_account'] == selected_account)
        ]

        st.markdown(f"#### {acc_data.get('name', 'Unknown')} (`{selected_account}`)")

        # Risk Drivers
        drivers = []
        mean_vel = acc_txs['velocity_sum_10tx'].mean() if len(acc_txs) > 0 else 0
        cross_border = acc_txs['cross_border_flag'].sum() if len(acc_txs) > 0 else 0
        mean_tx_count = acc_txs['tx_count_10'].mean() if len(acc_txs) > 0 else 0

        if mean_vel > global_vel_avg * 2:
            drivers.append(f"⚡ Transaction velocity is **{(mean_vel/max(global_vel_avg, 1)):.1f}×** above peer average")
        if cross_border > 0:
            drivers.append(f"🌐 **{int(cross_border)}** cross-border transactions (High Risk jurisdiction)")
        if 'receiver_gf_pagerank' in acc_txs.columns and len(acc_txs) > 0 and acc_txs['receiver_gf_pagerank'].max() >= 8:
            drivers.append("🕸️ Elevated network centrality (high-activity counterparties)")
        if 'sender_gf_out_degree' in acc_txs.columns and len(acc_txs) > 0 and acc_txs['sender_gf_out_degree'].max() >= 8:
            drivers.append("📤 High outgoing connectivity (potential Fan-Out)")
        if mean_tx_count > global_tx_count_avg * 2:
            drivers.append("📊 Transaction frequency is **anomalous** vs peers")
        if 'sender_gf_in_cycle' in acc_txs.columns and len(acc_txs) > 0 and acc_txs['sender_gf_in_cycle'].max() > 0:
            drivers.append("🔄 Part of circular transaction flow (cycle detected)")

        if not drivers:
            drivers.append("📋 Flagged via aggregate ML anomaly score")

        # Confidence and Momentum
        confidence_level = "High" if len(drivers) >= 3 else "Medium"
        confidence_reason = f"{len(drivers)} independent risk signals" if len(drivers) >= 3 else "Fewer corroborating indicators"

        tx_10 = acc_txs['tx_count_10'].mean() if len(acc_txs) > 0 else 0
        tx_30 = acc_txs['tx_count_30'].mean() if 'tx_count_30' in acc_txs.columns and len(acc_txs) > 0 else tx_10 * 3
        historical_20 = tx_30 - tx_10

        if historical_20 > 0 and tx_10 > (historical_20 / 2) * 1.5:
            momentum = "🔺 Rapidly Increasing"
        elif historical_20 > 0 and tx_10 < (historical_20 / 2) * 0.5:
            momentum = "🔻 Declining"
        else:
            momentum = "➡️ Stable"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Priority", acc_data['priority'])
        m2.metric("Risk Score", f"{acc_data['risk_recent_weighted']:.4f}")
        m3.metric("Confidence", confidence_level)
        m4.metric("Momentum", momentum)

        st.markdown("---")
        st.markdown("**Risk Drivers:**")
        for driver in drivers:
            st.markdown(f"  {driver}")

        # Typologies
        st.markdown("")
        st.markdown("**Triggered Typologies:**")
        typologies = []
        if cross_border > 0 and mean_vel > 500000:
            typologies.append("Cross-Border Burst")
        if 'sender_gf_out_degree' in acc_txs.columns and len(acc_txs) > 0 and acc_txs['sender_gf_out_degree'].max() >= 8:
            typologies.append("Fan-Out")
        if mean_tx_count > 10 and 'amount_local_npr' in acc_txs.columns and acc_txs['amount_local_npr'].mean() < 50000:
            typologies.append("Smurfing")
        if 'sender_gf_in_cycle' in acc_txs.columns and len(acc_txs) > 0 and acc_txs['sender_gf_in_cycle'].max() > 0:
            typologies.append("Circular Flow")

        if typologies:
            st.error("🚨 " + " | ".join(typologies))
        else:
            st.info("No specific typologies. Flagged via aggregate ML anomaly.")

        # Report Export
        st.markdown("---")
        if "Critical" in str(acc_data['priority']):
            action_rec = "🚨 Immediate review. Proceed to freeze pending RFI."
        elif "High" in str(acc_data['priority']):
            action_rec = "⚠️ Priority review within 24 hours."
        elif "Medium" in str(acc_data['priority']):
            action_rec = "📋 Routine review. Add to 30-day monitor."
        else:
            action_rec = "✅ No immediate action required."

        report_text = f"""# Case Investigation Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Account ID:** {selected_account}
**Account Name:** {acc_data.get('name', 'N/A')}
**Risk Score:** {acc_data['risk_recent_weighted']:.4f}
**Priority:** {acc_data['priority']}
**Model Confidence:** {confidence_level}
**Activity Momentum:** {momentum}

## Risk Drivers:
{chr(10).join('- ' + d for d in drivers)}

## Identified Typologies:
{', '.join(typologies) if typologies else 'None — flagged via aggregate ML anomaly'}

## Peer Comparison:
- Transaction Velocity: {mean_vel:,.0f} (Peer avg: {global_vel_avg:,.0f})
- Cross-Border Activity: {int(cross_border)} txs
- Transaction Count (10d): {mean_tx_count:.1f} (Peer avg: {global_tx_count_avg:.1f})

## Recommended Action:
{action_rec}
"""
        st.download_button(
            label="📄 Export Investigation Report",
            data=report_text,
            file_name=f"STR_Report_{selected_account}.md",
            mime="text/markdown"
        )

# ─────────────────────────────────────────────────────────────────────────
# TAB 2: DEEP INVESTIGATION
# ─────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("🔬 Deep Investigation Panel")

    col_inv1, col_inv2 = st.columns([1, 1])

    with col_inv1:
        st.subheader("Transaction History")
        inv_account = st.selectbox("Investigate Account:", acc_risk_df['account_id'].head(50), key='inv_acc')

        inv_txs = tx_df[
            (tx_df['Sender_account'] == inv_account) |
            (tx_df['Receiver_account'] == inv_account)
        ].sort_values('Date', ascending=False)

        if len(inv_txs) > 0:
            display_cols = ['Date', 'Sender_account', 'Receiver_account', 'amount_local_npr',
                            'cross_border_flag', 'tx_score']
            available_cols = [c for c in display_cols if c in inv_txs.columns]
            st.dataframe(inv_txs[available_cols].head(50), hide_index=True, use_container_width=True)

            # Transaction score distribution
            st.markdown("**Risk Score Distribution (this account's transactions):**")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.hist(inv_txs['tx_score'].values, bins=20, color='#FF4B4B', alpha=0.7, edgecolor='white')
            ax.axvline(x=0.5, color='orange', linestyle='--', label='Threshold')
            ax.set_xlabel('Transaction Risk Score')
            ax.set_ylabel('Count')
            ax.legend()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No transactions found for this account in the test set.")

    with col_inv2:
        st.subheader("Network Ego-Graph")
        st.markdown("*Visualizes 1-hop transaction neighborhood*")

        if len(inv_txs) > 0:
            try:
                from pyvis.network import Network
                import tempfile

                # Build ego graph
                net = Network(height="400px", width="100%", bgcolor="#0e1117", font_color="white",
                              directed=True)
                net.barnes_hut(gravity=-5000, central_gravity=0.3)

                # Add center node
                net.add_node(str(inv_account), label=str(inv_account)[:8], color='#FF4B4B',
                             size=30, title=f"Target: {inv_account}")

                # Add connected nodes
                counterparties = set()
                for _, row in inv_txs.iterrows():
                    sender = str(row['Sender_account'])
                    receiver = str(row['Receiver_account'])
                    if sender == str(inv_account):
                        counterparties.add(receiver)
                        if receiver not in [str(n['id']) for n in net.nodes]:
                            is_suspicious = receiver in [str(s) for s in suspicious_accs] if 'suspicious_accs' in dir() else False
                            color = '#FF6B6B' if is_suspicious else '#4ECDC4'
                            net.add_node(receiver, label=receiver[:8], color=color, size=15)
                        net.add_edge(str(inv_account), receiver,
                                     value=float(row.get('amount_local_npr', 1)) / 100000,
                                     title=f"NPR {row.get('amount_local_npr', 0):,.0f}")
                    else:
                        counterparties.add(sender)
                        if sender not in [str(n['id']) for n in net.nodes]:
                            is_suspicious = sender in [str(s) for s in suspicious_accs] if 'suspicious_accs' in dir() else False
                            color = '#FF6B6B' if is_suspicious else '#4ECDC4'
                            net.add_node(sender, label=sender[:8], color=color, size=15)
                        net.add_edge(sender, str(inv_account),
                                     value=float(row.get('amount_local_npr', 1)) / 100000,
                                     title=f"NPR {row.get('amount_local_npr', 0):,.0f}")

                st.markdown(f"**Counterparties:** {len(counterparties)} | **Transactions:** {len(inv_txs)}")

                # Save and display
                with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as f:
                    net.save_graph(f.name)
                    with open(f.name, 'r') as html_file:
                        html_content = html_file.read()
                    st.components.v1.html(html_content, height=420)
                    os.unlink(f.name)

            except ImportError:
                st.warning("PyVis not installed. Install with: `pip install pyvis`")
            except Exception as e:
                st.warning(f"Network visualization error: {e}")

        # Feature Importance (Global)
        st.subheader("Top Feature Importances")
        if feature_importances is not None and len(feature_importances) > 0:
            top_features = feature_importances.head(15)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.barh(top_features['feature'].values[::-1],
                     top_features['importance'].values[::-1],
                     color='#667eea')
            ax2.set_xlabel('Gain')
            ax2.set_title('Top 15 Features (LightGBM)')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────
# TAB 3: NLP FEATURES
# ─────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🧠 Entity-Faithful STR Summarization")
    st.markdown("""
    This module demonstrates Track 6 capabilities: automated summarization of 
    Suspicious Transaction Report (STR) narratives using open-source local LLMs,
    with entity-faithfulness validation.
    """)

    sample_narrative = """Account A5 has received 15 incoming transfers under threshold within 24h, totalling NPR 2,300,000. Customer could not explain the origin of funds. The funds were subsequently wired out to an overseas account at Nexus Corp on 2025-06-14."""

    col_nlp1, col_nlp2 = st.columns([1, 1])

    with col_nlp1:
        narrative_input = st.text_area("Input Raw STR Narrative:", value=sample_narrative, height=150)
        risk_score_input = st.number_input("Inject Risk Score:", min_value=0.0, max_value=1.0, value=0.92)
        typology_input = st.text_input("Inject Typology:", value="Smurfing")

        if st.button("🔬 Generate Summary", type="primary"):
            with st.spinner("Running NLP Pipeline (NER + LLM)..."):
                summarizer = LocalLLMSummarizer(use_mock=True)
                summary, entities = process_str_narrative(
                    narrative_input, risk_score_input, typology_input, summarizer
                )
                validation = validate_summary(narrative_input, summary, entities)

                st.session_state['nlp_result'] = {
                    'summary': summary,
                    'entities': entities,
                    'validation': validation
                }

    with col_nlp2:
        if 'nlp_result' in st.session_state:
            result = st.session_state['nlp_result']

            st.subheader("Generated Summary")
            st.info(result['summary'])

            st.subheader("Faithfulness Validation")
            score = result['validation']['faithfulness_score']
            grade = result['validation']['grade']

            m1, m2 = st.columns(2)
            m1.metric("Faithfulness Score", f"{score:.1f}%")
            m2.metric("Grade", grade)

            st.markdown("**Extracted Entities:**")
            for cat, ents in result['entities'].items():
                if ents:
                    st.markdown(f"- **{cat.title()}:** {', '.join(ents)}")

            if not result['validation']['is_fully_faithful']:
                details = result['validation']['details']
                missing_all = []
                for cat in details:
                    missing_all.extend(details[cat].get('missing', []))
                if missing_all:
                    st.error(f"⚠️ Missing entities in summary: {', '.join(missing_all)}")
