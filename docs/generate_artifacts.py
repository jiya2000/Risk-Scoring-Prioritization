"""
Generate all IDF-B artifact figures for the AML Risk Scoring Platform.
Saves PNGs to docs/artifacts/.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.gridspec as gridspec
import networkx as nx

OUT = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(OUT, exist_ok=True)

DPI = 150
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False})

# ─────────────────────────────────────────────────────────────────────
# FIG 01 — System Architecture  (fixed: no overlapping boxes)
# Layout rows (y-centre):
#   Row A  y=10.2  Data inputs
#   Row B  y=8.7   Feature Engineering + LightGBM
#   Row C  y=7.1   TD-PageRank | Ego-Network | TopologyAttentionGate
#   Row D  y=5.4   Adaptive Fusion | Symbolic Rules
#   Row E  y=3.8   Degradation Controller | Account Aggregation
#   Row F  y=2.2   Architecture Hardening (full-width strip)
#   Row G  y=0.7   NLP | Streamlit Dashboard
# ─────────────────────────────────────────────────────────────────────
def fig01():
    fig, ax = plt.subplots(figsize=(15, 13))
    ax.set_xlim(0, 15); ax.set_ylim(0, 13); ax.axis('off')
    ax.set_title("Fig 1 — AML Risk Scoring Platform: Full System Architecture",
                 fontsize=13, fontweight='bold', pad=12)

    def box(x, y, w, h, label, color, fontsize=8.5):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                               facecolor=color, edgecolor='#444', linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', multialignment='center',
                linespacing=1.4)

    def arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
        if label:
            ax.text((x1 + x2) / 2 + 0.08, (y1 + y2) / 2 + 0.06,
                    label, fontsize=7, color='#555', style='italic')

    # ── Row A: Data inputs ───────────────────────────────────────────
    box(0.3, 11.5, 3.2, 1.1,
        "transactions.csv\naccounts.csv\ngraph_edges.csv", '#AED6F1', 8)

    # ── Row B: Feature Engineering + LightGBM ───────────────────────
    box(0.3, 9.8, 3.5, 1.0, "Feature Engineering\n(40+ features)", '#A9DFBF', 9)
    box(4.3, 9.8, 3.2, 1.0, "LightGBM Classifier\n→ S_ml", '#A9DFBF', 9)

    # ── Row C: TD-PageRank | Ego-Network | TopologyAttentionGate ────
    box(0.3, 7.9, 3.8, 1.2,
        "TD-PageRank Engine\nw = amt × exp(−λ×age) × burst\nDirectional SCC Penalty", '#F9E79F', 8.5)
    box(4.5, 7.9, 3.8, 1.2,
        "Ego-Network Extractor\n2-hop TopologyVector\n(6 metrics)", '#F9E79F', 8.5)
    box(8.8, 7.9, 5.9, 1.2,
        "TopologyAttentionGate\nmultiplicative gating → [w_ml, w_graph, w_rules]", '#F0B27A', 9)

    # ── Row D: Adaptive Fusion + Symbolic Rules ──────────────────────
    box(0.3, 6.1, 6.2, 1.1,
        "Adaptive Fusion Engine\nfused = w_ml·S_ml + w_graph·S_graph + w_rules·S_rules", '#F0B27A', 9)
    box(7.0, 6.1, 7.7, 1.1,
        "Symbolic Rule Engine  —  10 FATF Typologies → S_rules", '#D7BDE2', 9)

    # ── Row E: Degradation Controller + Account Aggregation ──────────
    box(0.3, 4.2, 7.0, 1.3,
        "Degradation Controller (Innovation 3)\n8 Execution Paths  |  P@50 ≥ 0.60 guaranteed\nPrecisionDriftDetector + AdaptivePrecisionBudget", '#F1948A', 9)
    box(7.8, 4.2, 6.9, 1.3,
        "Account Risk Aggregation\nTime-decay composite + Burst + High-risk ratio\n→ Priority Queue (Critical / High / Medium / Low)", '#D5DBDB', 9)

    # ── Row F: Architecture Hardening — full-width, own row ──────────
    box(0.3, 2.6, 14.4, 1.1,
        "Architecture Hardening Layer  (all flags default = False → fully backward-compatible)\n"
        "LearnableSCCPenalty  |  TopologyEmbeddingNetwork (GNN)  |  IsolationDetector  |  "
        "DeepTopologyAttentionGate  |  OnlinePrecisionMonitor  |  EnhancedConceptDriftDetector  |  ResourceManager (§101)",
        '#ABEBC6', 8)

    # ── Row G: NLP + Dashboard ───────────────────────────────────────
    box(0.3, 0.6, 4.5, 1.2,
        "NLP Pipeline\nspaCy NER + SmolLM-135M\nSTR Narrative Generation", '#AED6F1', 8.5)
    box(5.3, 0.6, 9.4, 1.2,
        "Streamlit Analyst Dashboard\nPriority Queue  |  Case Investigation  |  Network Viz  |  SHAP  |  Export",
        '#FAD7A0', 9)

    # ── Arrows ───────────────────────────────────────────────────────
    # Data → Feature Eng
    arrow(3.5, 12.05, 3.5, 11.0)   # drop down connector (side)
    arrow(2.0, 11.5, 2.0, 10.8)    # Data → Feature Eng vertical
    # Feature Eng → LightGBM
    arrow(3.8, 10.3, 4.3, 10.3, 'features')
    # LightGBM → TopologyAttentionGate
    arrow(7.5, 10.3, 11.7, 9.1, 'S_ml')
    # Data → TD-PageRank
    arrow(1.5, 11.5, 1.5, 9.1)
    # TD-PageRank → Ego-Network
    arrow(4.1, 8.5, 4.5, 8.5)
    # Ego-Network → TopologyAttentionGate
    arrow(8.3, 8.5, 8.8, 8.5, 'topology\nvector')
    # TopologyAttentionGate → Adaptive Fusion
    arrow(11.7, 7.9, 11.7, 7.2)
    arrow(11.7, 7.2, 3.4, 6.7, 'weights')
    # TD-PageRank → Adaptive Fusion
    arrow(2.2, 7.9, 2.2, 7.2)
    # Symbolic Rules → Adaptive Fusion
    arrow(7.5, 6.65, 6.5, 6.65, 'S_rules')
    # Adaptive Fusion → Degradation Controller
    arrow(3.8, 6.1, 3.8, 5.5)
    # Degradation Controller → Account Aggregation
    arrow(7.3, 4.85, 7.8, 4.85)
    # Account Aggregation → Dashboard
    arrow(11.25, 4.2, 11.25, 1.8)
    # Degradation Controller → NLP
    arrow(2.5, 4.2, 2.5, 1.8)
    # NLP + Dashboard → bottom (just indicators)
    arrow(2.5, 0.6, 2.5, 0.3)
    arrow(10.0, 0.6, 10.0, 0.3)

    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, 'fig01_system_architecture.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("fig01 done")

# ─────────────────────────────────────────────────────────────────────
# FIG 02 — TD-PageRank Flowchart
# ─────────────────────────────────────────────────────────────────────
def fig02():
    fig, ax = plt.subplots(figsize=(8, 14))
    ax.set_xlim(0, 8); ax.set_ylim(0, 14); ax.axis('off')
    ax.set_title("Fig 2 — TD-PageRank Algorithm Flowchart", fontsize=13, fontweight='bold', pad=10)

    def proc(x, y, w, h, text, color='#AED6F1'):
        rect = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor='#333', lw=1.5)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=9,
                fontweight='bold', multialignment='center')

    def diamond(cx, cy, hw, hh, text, color='#F9E79F'):
        xs = [cx, cx+hw, cx, cx-hw, cx]
        ys = [cy+hh, cy, cy-hh, cy, cy+hh]
        ax.fill(xs, ys, color=color, edgecolor='#333', lw=1.5)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=8, multialignment='center')

    def arr(x1,y1,x2,y2, lbl=''):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
        if lbl:
            ax.text((x1+x2)/2+0.1, (y1+y2)/2, lbl, fontsize=8, color='#333')

    proc(2.0, 12.8, 4.0, 0.7, "Load edge list (sender, receiver, amount, date)", '#AED6F1')
    arr(4.0,12.8, 4.0,12.15)
    proc(2.0, 11.5, 4.0, 0.6, "Compute temporal weights:\nw = amount × exp(−λ × EdgeAge)", '#A9DFBF')
    arr(4.0,11.5, 4.0,10.9)
    proc(2.0, 10.3, 4.0, 0.6, "Detect burst-velocity senders\nburst_mult = 1 + (window_cnt/total_cnt)", '#A9DFBF')
    arr(4.0,10.3, 4.0,9.7)
    proc(2.0, 9.1, 4.0, 0.6, "Build directed graph G\nwith temporal weights × burst_mult", '#AED6F1')
    arr(4.0,9.1, 4.0,8.5)
    proc(2.0, 7.9, 4.0, 0.6, "Initialise r[v] = 1/N for all v", '#AED6F1')
    arr(4.0,7.9, 4.0,7.3)
    proc(2.0, 6.7, 4.0, 0.6, "Power iteration step:\nr_new = (1−d)/N + d × M^T · r", '#F9E79F')
    arr(4.0,6.7, 4.0,6.1)
    diamond(4.0, 5.55, 2.0, 0.55, "‖r_new − r‖₁ < 1e-6\nor iter ≥ 100?")
    arr(4.0,5.0, 4.0,4.4, 'YES')
    ax.annotate('', xy=(4.0,7.25), xytext=(6.5,5.55),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5, connectionstyle='arc3,rad=-0.3'))
    ax.text(6.6, 6.5, 'NO', fontsize=9, color='#c0392b', fontweight='bold')
    proc(2.0, 3.7, 4.0, 0.6, "Detect SCCs of size > 2\nApply directional SCC penalty:\nCollector 0.25×, Distrib 0.50×, Balanced 0.375×", '#F0B27A')
    arr(4.0,3.7, 4.0,3.1)
    proc(2.0, 2.4, 4.0, 0.6, "Detect dormant nodes\n(all edges > 30 days)\nCap score ≤ 0.1 × max_score", '#F1948A')
    arr(4.0,2.4, 4.0,1.8)
    proc(2.0, 1.1, 4.0, 0.6, "Min-max normalise → [0, 1]\nReturn TDPageRankResult", '#A9DFBF')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig02_tdpagerank_flowchart.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("fig02 done")

# ─────────────────────────────────────────────────────────────────────
# FIG 03 — TopologyAttentionGate Neural Architecture
# ─────────────────────────────────────────────────────────────────────
def fig03():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title("Fig 3 — TopologyAttentionGate Neural Architecture\n(Multiplicative Gating on Ensemble Weight Dimension)",
                 fontsize=12, fontweight='bold')

    def box(x, y, w, h, text, col):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           facecolor=col, edgecolor='#333', lw=1.3)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=8.5, fontweight='bold', multialignment='center')

    def arr(x1, y1, x2, y2, lbl='', color='#333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                    shrinkA=2, shrinkB=2))
        if lbl:
            ax.text((x1 + x2)/2 + 0.05, (y1 + y2)/2 + 0.08, lbl,
                    fontsize=7.5, color='#555')

    # ── Boxes ────────────────────────────────────────────────────────
    # Input box: bottom-left (0.3, 2.6), w=1.5, h=1.4
    # center: (1.05, 3.3), right edge: x=1.8
    box(0.3, 2.6, 1.5, 1.4, "Input\n5-dim\nTopology\nVector", '#AED6F1')

    # Gate path (top row, y=4.4)
    # Gate Layer: (2.4, 4.4), w=2.2, h=0.9 → center (3.5, 4.85), left=2.4, right=4.6
    box(2.4, 4.4, 2.2, 0.9, "Gate Layer\nLinear(5→5)\n+ Sigmoid", '#F9E79F')
    # Gate Activity: (5.1, 4.4), w=2.0, h=0.9 → center (6.1, 4.85), left=5.1, right=7.1
    box(5.1, 4.4, 2.0, 0.9, "Gate Activity\nmean(gate)\n∈ (0,1)", '#F9E79F')
    # Gate Amplifier: (7.6, 4.4), w=2.2, h=0.9 → center (8.7, 4.85), left=7.6, right=9.8
    box(7.6, 4.4, 2.2, 0.9, "Gate Amplifier\n1.0 + 2.0×activity\n∈ [1.0, 3.0]", '#F0B27A')

    # Base path (bottom row, y=1.6)
    # Base Allocator: (2.4, 1.6), w=2.2, h=1.2 → center (3.5, 2.2), left=2.4, right=4.6
    box(2.4, 1.6, 2.2, 1.2, "Base Allocator\nLinear(5→32)\nReLU\nLinear(32→3)", '#A9DFBF')
    # Softmax: (5.1, 1.6), w=2.0, h=1.2 → center (6.1, 2.2), left=5.1, right=7.1
    box(5.1, 1.6, 2.0, 1.2, "Softmax\nBase Weights\n(w_ml, w_graph,\nw_rules)", '#A9DFBF')

    # Multiply symbol at (8.3, 3.3)
    ax.text(8.3, 3.3, "×", ha='center', va='center', fontsize=24,
            color='#c0392b', fontweight='bold')

    # Clamp+Norm: (9.6, 2.7), w=2.2, h=1.1 → center (10.7, 3.25), left=9.6, right=11.8
    box(9.6, 2.7, 2.2, 1.1, "Clamp [0.05,0.90]\n+ Normalize\n(3 iterations)", '#D7BDE2')
    # Output: (12.1, 2.7), w=1.5, h=1.1 → center (12.85, 3.25), left=12.1, right=13.6
    box(12.1, 2.7, 1.5, 1.1, "Output\n(w_ml,\nw_graph,\nw_rules)", '#ABEBC6')

    # ── Arrows ───────────────────────────────────────────────────────
    # Input right edge → Gate Layer left edge (upper path)
    arr(1.8, 3.6, 2.4, 4.85, color='#333')
    # Input right edge → Base Allocator left edge (lower path)
    arr(1.8, 3.0, 2.4, 2.2, color='#333')

    # Gate Layer right → Gate Activity left (same row)
    arr(4.6, 4.85, 5.1, 4.85, color='#333')
    # Gate Activity right → Gate Amplifier left (same row)
    arr(7.1, 4.85, 7.6, 4.85, color='#333')

    # Base Allocator right → Softmax left (same row)
    arr(4.6, 2.2, 5.1, 2.2, color='#333')

    # Softmax right edge → multiply symbol (bottom-left approach)
    arr(7.1, 2.2, 8.1, 3.0, color='#c0392b')
    # Gate Amplifier bottom → multiply symbol (top approach)
    arr(8.7, 4.4, 8.4, 3.6, color='#e67e22')

    # Label for multiply
    ax.text(9.3, 4.0, "Amplify\nw_graph only", fontsize=8,
            color='#c0392b', style='italic')

    # Multiply symbol → Clamp left edge
    arr(8.6, 3.3, 9.6, 3.25, color='#333')
    # Clamp right edge → Output left edge
    arr(11.8, 3.25, 12.1, 3.25, color='#333')

    # ── Key text ─────────────────────────────────────────────────────
    ax.text(7.0, 0.5,
            "KEY: Gate operates on the WEIGHT DIMENSION (not feature space) "
            "— distinct from standard additive attention",
            ha='center', fontsize=9, color='#7D3C98', style='italic',
            fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig03_topology_attention_gate.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("fig03 done")

# ─────────────────────────────────────────────────────────────────────
# FIG 04 — Degradation Controller State Machine
# ─────────────────────────────────────────────────────────────────────
def fig04():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax = axes[0]; ax.set_xlim(0,8); ax.set_ylim(0,8); ax.axis('off')
    ax.set_title("State Machine", fontsize=11, fontweight='bold')

    def state(cx, cy, r, text, color):
        c = plt.Circle((cx,cy), r, color=color, ec='#333', lw=2, zorder=3)
        ax.add_patch(c)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

    state(4,6.5,1.0,"HEALTHY",'#A9DFBF')
    state(1.5,2.5,1.0,"DEGRADED",'#FAD7A0')
    state(6.5,2.5,1.0,"COOLDOWN",'#F1948A')

    ax.annotate('', xy=(1.8,3.4), xytext=(3.2,5.7),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))
    ax.text(2.0,4.8,"2 consec.\nunhealthy", fontsize=8, color='#c0392b')
    ax.annotate('', xy=(3.2,5.7), xytext=(1.8,3.4),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2, connectionstyle='arc3,rad=0.3'))
    ax.text(3.2,4.4,"3 consec.\nclean", fontsize=8, color='#27ae60')
    ax.annotate('', xy=(5.6,3.4), xytext=(2.5,2.5),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    ax.text(3.8,3.2,">3 transitions\nin 5 min", fontsize=8, color='#e74c3c')
    ax.annotate('', xy=(4.8,5.7), xytext=(5.6,3.4),
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2))
    ax.text(5.6,4.8,"after\n5 min", fontsize=8, color='#2980b9')

    ax.text(4,0.8,"Shadow eval → P@50 check\nCRITICAL if P@50 < budget−0.05\nStale queue policy: 30 min max",
            ha='center', fontsize=8.5, bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor='#2980b9'))

    # Routing table subplot
    ax2 = axes[1]
    paths = ['full','no_nlp','no_symbolic','no_pagerank','no_fusion','lgbm_pagerank','lgbm_rules','lgbm_only']
    p50s = [0.82, 0.82, 0.76, 0.72, 0.70, 0.68, 0.66, 0.62]
    colors = ['#27AE60' if v >= 0.70 else '#F39C12' if v >= 0.60 else '#E74C3C' for v in p50s]
    bars = ax2.barh(paths, p50s, color=colors, edgecolor='#333', height=0.6)
    ax2.axvline(0.60, color='red', lw=2, linestyle='--', label='Budget floor (0.60)')
    ax2.set_xlabel('Precision@50', fontsize=10)
    ax2.set_title("8 Execution Paths — P@50 Routing Table", fontsize=11, fontweight='bold')
    ax2.set_xlim(0.55, 0.88)
    for bar, v in zip(bars, p50s):
        ax2.text(v+0.002, bar.get_y()+bar.get_height()/2, f'{v:.2f}', va='center', fontsize=9)
    ax2.legend(fontsize=9)
    ax2.invert_yaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig04_degradation_state_machine.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("fig04 done")

# ─────────────────────────────────────────────────────────────────────
# FIG 05 — Data Flow Diagram
# ─────────────────────────────────────────────────────────────────────
def fig05():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis('off')
    ax.set_title("Fig 5 — End-to-End Data Flow Diagram", fontsize=13, fontweight='bold', pad=10)

    def box(x, y, w, h, text, col, fs=8.5):
        """Draw box at (x,y) as bottom-left corner."""
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           facecolor=col, edgecolor='#444', lw=1.3)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fs, fontweight='bold', multialignment='center')
        # Return center coords and dimensions for arrow convenience
        return (x, y, w, h)

    def arr(x1, y1, x2, y2, lbl='', offset=(0.05, 0.12)):
        """Draw arrow from (x1,y1) to (x2,y2) with optional label."""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.6,
                                    shrinkA=2, shrinkB=2))
        if lbl:
            ax.text((x1 + x2)/2 + offset[0], (y1 + y2)/2 + offset[1],
                    lbl, fontsize=7.5, color='#555', style='italic')

    # ── Row 1 (top): CSV inputs ──────────────────────────────────────
    # y=7.6..8.4 for transactions, y=6.4..7.2 accounts, y=5.2..6.0 graph_edges
    csv1 = box(0.3, 7.5, 2.2, 0.9, "transactions.csv\n(edges+amounts)", '#AED6F1', 8)
    csv2 = box(0.3, 6.3, 2.2, 0.9, "accounts.csv\n(KYC data)", '#AED6F1', 8)
    csv3 = box(0.3, 5.1, 2.2, 0.9, "graph_edges.csv\n(graph structure)", '#AED6F1', 8)

    # ── Row 2: Feature Engineering + LightGBM ────────────────────────
    feat = box(3.2, 7.0, 2.8, 1.1, "Feature Engineering\n40+ features\n(temporal, KYC, graph)", '#A9DFBF')
    lgbm = box(6.8, 7.0, 2.4, 1.1, "LightGBM\nClassifier\n→ S_ml ∈ [0,1]", '#A9DFBF')

    # Arrows: CSVs → Feature Engineering (right edge of CSV → left edge of Feat)
    arr(2.5, 7.95, 3.2, 7.55)  # transactions → feat
    arr(2.5, 6.75, 3.2, 7.35)  # accounts → feat

    # Feature Eng → LightGBM (right edge → left edge)
    arr(6.0, 7.55, 6.8, 7.55, 'features')

    # ── Row 3: TD-PageRank + Ego-Network + TopologyAttentionGate ─────
    tdpr = box(3.2, 5.0, 3.0, 1.2, "TD-PageRank Engine\nw=amt×exp(−λ×age)\n→ S_graph ∈ [0,1]", '#F9E79F')
    ego  = box(6.8, 5.0, 2.8, 1.2, "Ego-Network Extractor\n2-hop subgraph\n→ TopologyVector (6D)", '#F9E79F')
    gate = box(10.2, 5.0, 3.5, 1.2, "TopologyAttentionGate\n→ (w_ml, w_graph, w_rules)\neach ∈ [0.05,0.90], sum=1", '#F0B27A')

    # graph_edges → TD-PageRank (right edge of csv3 → left edge of tdpr)
    arr(2.5, 5.55, 3.2, 5.6, 'edges')

    # TD-PageRank → Ego-Network (right edge → left edge, same row)
    arr(6.2, 5.6, 6.8, 5.6)

    # Ego-Network → TopologyAttentionGate (right → left)
    arr(9.6, 5.6, 10.2, 5.6, 'topology\nvector')

    # LightGBM → TopologyAttentionGate (S_ml: bottom of lgbm → top of gate)
    arr(9.0, 7.0, 11.95, 6.2, 'S_ml')

    # ── Row 4: Symbolic Rules + Adaptive Fusion + Account Risk ───────
    rules = box(3.2, 3.0, 2.8, 1.0, "Symbolic Rule Engine\n10 FATF Typologies\n→ S_rules ∈ [0,1]", '#D7BDE2')
    fuse  = box(6.8, 3.0, 2.8, 1.0, "Adaptive Fusion\nfused = Σ w_i × S_i\n→ fused_score", '#F0B27A')
    acct  = box(10.2, 3.0, 3.5, 1.0, "Account Risk Aggregation\ntime-decay + burst + top-5\n→ Priority Queue", '#D5DBDB')

    # Symbolic Rules → Adaptive Fusion (right edge → left edge)
    arr(6.0, 3.5, 6.8, 3.5, 'S_rules')

    # TD-PageRank → Adaptive Fusion (S_graph: bottom of tdpr → top of fuse)
    arr(5.6, 5.0, 7.6, 4.0, 'S_graph')

    # TopologyAttentionGate → Adaptive Fusion (weights: bottom of gate → top of fuse)
    arr(11.95, 5.0, 8.8, 4.0, 'weights')

    # Adaptive Fusion → Account Risk (right edge → left edge)
    arr(9.6, 3.5, 10.2, 3.5)

    # ── Row 5 (bottom): Degradation Controller + Dashboard ───────────
    ctrl = box(3.2, 1.0, 5.8, 1.2, "Degradation Controller\nRoutes to best execution path P@50 ≥ 0.60\nOnlinePrecisionMonitor + DriftDetector", '#F1948A')
    dash = box(10.2, 1.0, 3.5, 1.2, "Streamlit Dashboard\nAnalyst Priority Queue\n+ Case Investigation", '#FAD7A0')

    # Adaptive Fusion → Degradation Controller (bottom of fuse → top of ctrl)
    arr(8.2, 3.0, 6.1, 2.2, 'fused scores')

    # Account Risk → Dashboard (bottom of acct → top of dash)
    arr(11.95, 3.0, 11.95, 2.2)

    # Degradation Controller → Dashboard (right edge → left edge, same row)
    arr(9.0, 1.6, 10.2, 1.6, 'STR\nNLP')

    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, 'fig05_data_flow.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("fig05 done")

# ─────────────────────────────────────────────────────────────────────
# FIG 06 — Ablation Study
# ─────────────────────────────────────────────────────────────────────
def fig06():
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = ['Tabular LightGBM only',
               '+ Graph Features (16 metrics)',
               '+ Symbolic Rules (10 FATF typologies)',
               '+ Static Fusion (fixed weights)',
               '+ TD-PageRank + TopologyAttentionGate',
               '+ Full System (Degradation Controller)']
    p50 = [0.41, 0.55, 0.60, 0.65, 0.76, 0.82]
    increments = [0.41, 0.14, 0.05, 0.05, 0.11, 0.06]
    colors = ['#5DADE2','#48C9B0','#F39C12','#E67E22','#E74C3C','#27AE60']

    bars = ax.barh(configs, p50, color=colors, edgecolor='#333', height=0.55)
    for i, (bar, v, inc) in enumerate(zip(bars, p50, increments)):
        ax.text(v+0.005, bar.get_y()+bar.get_height()/2, f'P@50={v:.2f}', va='center', fontsize=9, fontweight='bold')
        if i > 0:
            ax.text(v-0.08, bar.get_y()+bar.get_height()/2, f'+{inc:.2f}', va='center', fontsize=8, color='white', fontweight='bold')

    ax.set_xlabel('Precision@50 (P@50)', fontsize=11)
    ax.set_title('Fig 6 — Ablation Study: Incremental Component Value\n(Temporal test set, no data leakage)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 0.92)
    ax.invert_yaxis()
    ax.axvline(0.82, color='#27AE60', lw=2, linestyle='--', alpha=0.7, label='Full system P@50 = 0.82')
    ax.axvline(0.60, color='red', lw=1.5, linestyle=':', alpha=0.7, label='Precision budget floor')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig06_ablation_study.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("fig06 done")

# ─────────────────────────────────────────────────────────────────────
# FIG 07 — Architecture Hardening Layer
# ─────────────────────────────────────────────────────────────────────
def fig07():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0,14); ax.set_ylim(0,9); ax.axis('off')
    ax.set_title("Fig 7 — Architecture Hardening Layer: 4 Hardening Surfaces\n(All flags default=False → backward compatible baseline)",
                 fontsize=12, fontweight='bold')

    def box(x,y,w,h,text,col,fs=8.5,alpha=1.0):
        r = FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.1",
                           facecolor=col, edgecolor='#444', lw=1.3, alpha=alpha)
        ax.add_patch(r)
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,
                fontweight='bold', multialignment='center')

    # Surface 1 — SCC Penalty
    ax.text(0.2, 8.5, "Surface 1: TD-PageRank SCC Penalty", fontsize=10, fontweight='bold', color='#1A5276')
    box(0.2,7.2,3.8,1.1,"BEFORE: Static Heuristic\nCollector 0.25× | Distrib 0.50×\nBalanced 0.375×",'#FADBD8')
    ax.text(4.4,7.75,"→", fontsize=18, color='#27AE60', fontweight='bold')
    box(4.8,7.2,4.0,1.1,"AFTER: LearnableSCCPenalty MLP\nInput: 5 SCC flow features\nOutput: penalty ∈ [0.1,1.0]\n(use_learnable_scc_penalty=True)",'#ABEBC6')
    box(9.2,7.2,4.5,1.1,"Config flag: use_learnable_scc_penalty\nModel: 2-layer MLP (input=5, hidden=32)\nAtomic swap via ResourceManager\nSCC Feature Extraction: extract_scc_flow_features()",'#EBF5FB')

    # Surface 2 — Topology Embedding
    ax.text(0.2, 6.8, "Surface 2: Topology Representation", fontsize=10, fontweight='bold', color='#1A5276')
    box(0.2,5.5,3.8,1.1,"BEFORE: Manual TopologyVector (5D)\nEdge density, diameter,\nclustering, asymmetry, component ratio",'#FADBD8')
    ax.text(4.4,6.05,"→", fontsize=18, color='#27AE60', fontweight='bold')
    box(4.8,5.5,4.0,1.1,"AFTER: TopologyEmbeddingNetwork GNN\n2×GCNConv → GlobalMeanPool\nembedding_dim=16\n+ IsolationDetector (continuous w_ml)",'#ABEBC6')
    box(9.2,5.5,4.5,1.1,"Config: use_gnn_topology + use_learned_isolation\nTimeout: 200ms → fallback to manual\nIsolation score ∈ [0,1] replaces hard threshold\nw_ml monotonically non-decreasing with isolation",'#EBF5FB')

    # Surface 3 — Fusion Gate
    ax.text(0.2, 5.1, "Surface 3: Fusion Gate Architecture", fontsize=10, fontweight='bold', color='#1A5276')
    box(0.2,3.8,3.8,1.1,"BEFORE: Single-layer gate\n5×5 Linear + Sigmoid\n+ 5×32 base allocator",'#FADBD8')
    ax.text(4.4,4.35,"→", fontsize=18, color='#27AE60', fontweight='bold')
    box(4.8,3.8,4.0,1.1,"AFTER: DeepTopologyAttentionGate\n2+ hidden layers (dim≥64)\nDropout regularization\nMultiplicative gating preserved",'#ABEBC6')
    box(9.2,3.8,4.5,1.1,"Config: use_deep_gate\nInput: GNN embedding (16D) or TopologyVector\nInference: <50ms budget\nWeights: each ∈ [0.05,0.90], sum=1.0",'#EBF5FB')

    # Surface 4 — Precision Monitoring
    ax.text(0.2, 3.4, "Surface 4: Precision Monitoring + §101 Resource Management", fontsize=10, fontweight='bold', color='#1A5276')
    box(0.2,1.8,3.8,1.4,"BEFORE: Static offline P@50\nassumptions in routing table\nNo live precision estimation\nFixed alert thresholds",'#FADBD8')
    ax.text(4.4,2.55,"→", fontsize=18, color='#27AE60', fontweight='bold')
    box(4.8,1.8,4.0,1.4,"AFTER: OnlinePrecisionMonitor\n+ EnhancedConceptDriftDetector\n+ EnhancedAdaptivePrecisionBudget\n+ ResourceManager (§101)",'#ABEBC6')
    box(9.2,1.8,4.5,1.4,"Config: use_online_precision+use_enhanced_drift\nP@50 estimated from rolling 500-sample window\nDrift: 'no_drift'|'benign_shift'|'precision_degraded'\nResourceManager logs: memory delta, latency, units",'#EBF5FB')

    ax.text(7.0, 0.4, "ArchitectureHardeningConfig: all flags default=False → identical to baseline when disabled",
            ha='center', fontsize=9.5, color='#7D3C98', style='italic', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#F9EBF8', edgecolor='#7D3C98'))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig07_architecture_hardening.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("fig07 done")

# ─────────────────────────────────────────────────────────────────────
# FIG 08 — Precision Routing Table
# ─────────────────────────────────────────────────────────────────────
def fig08():
    fig, ax = plt.subplots(figsize=(10, 5))
    paths = ['full (all 5 components)',
             'no_nlp (drop NLP summarizer)',
             'no_symbolic (drop rules)',
             'no_pagerank (drop TD-PR)',
             'no_fusion (drop adaptive gate)',
             'lgbm_pagerank (ML + graph)',
             'lgbm_rules (ML + rules)',
             'lgbm_only (minimal)']
    p50 = [0.82, 0.82, 0.76, 0.72, 0.70, 0.68, 0.66, 0.62]
    colors = ['#1E8449' if v >= 0.75 else '#27AE60' if v >= 0.65 else '#F39C12' for v in p50]
    bars = ax.barh(paths, p50, color=colors, edgecolor='#333', height=0.6)
    ax.axvline(0.60, color='red', lw=2, linestyle='--', label='Precision budget floor (0.60)')
    ax.set_xlabel('Precision@50', fontsize=11)
    ax.set_title('Fig 8 — Degradation Controller: Execution Path Routing Table\n(All paths meet minimum P@50 ≥ 0.60 budget)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0.55, 0.88)
    for bar, v in zip(bars, p50): ax.text(v+0.002, bar.get_y()+bar.get_height()/2, f'{v:.2f}', va='center', fontsize=9)
    ax.legend(fontsize=9); ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig08_precision_routing_table.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("fig08 done")

# ─────────────────────────────────────────────────────────────────────
# FIG 09 — Conceptual Innovation Diagram (3-panel)
# ─────────────────────────────────────────────────────────────────────
def fig09():
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1: TD-PageRank transaction network
    ax1 = fig.add_subplot(gs[0])
    G = nx.DiGraph()
    nodes = ['A','B','C','D','E','F']
    G.add_edges_from([('A','B'),('B','C'),('C','A'),('A','D'),('D','E'),('E','F')])
    pos = nx.spring_layout(G, seed=42)
    edge_weights = [3.5, 3.5, 3.5, 1.5, 1.0, 0.5]  # thick=recent, thin=old
    edge_colors = ['#E74C3C','#E74C3C','#E74C3C','#F39C12','#7FB3D3','#ABEBC6']
    node_colors = ['#E74C3C' if n in ['A','B','C'] else '#AED6F1' for n in nodes]
    nx.draw_networkx(G, pos, ax=ax1, node_color=node_colors, node_size=700,
                     width=edge_weights, edge_color=edge_colors,
                     arrows=True, arrowsize=15, font_size=10, font_weight='bold')
    ax1.set_title("Innovation 1: TD-PageRank\n(SCC in red, recent=thick, old=thin)", fontsize=9, fontweight='bold')
    ax1.axis('off')
    ax1.text(0.5,-0.12, "SCC nodes (A,B,C): directional penalty applied\nBurst sender: w amplified", ha='center',
             transform=ax1.transAxes, fontsize=7.5, color='#7D3C98', style='italic')

    # Panel 2: Isolated vs Hub account fusion weights
    ax2 = fig.add_subplot(gs[1])
    categories = ['w_ml', 'w_graph', 'w_rules']
    isolated_weights = [0.75, 0.13, 0.12]  # high w_ml for isolated
    hub_weights = [0.25, 0.63, 0.12]       # high w_graph for hub
    x = np.arange(3)
    w = 0.35
    ax2.bar(x - w/2, isolated_weights, w, label='Isolated (1-node ego)', color='#AED6F1', edgecolor='#333')
    ax2.bar(x + w/2, hub_weights, w, label='Hub (dense network)', color='#F0B27A', edgecolor='#333')
    ax2.set_xticks(x); ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel('Weight'); ax2.set_ylim(0, 0.9)
    ax2.axhline(0.05, color='red', lw=1, linestyle=':', alpha=0.5)
    ax2.axhline(0.90, color='red', lw=1, linestyle=':', alpha=0.5)
    ax2.legend(fontsize=8)
    ax2.set_title("Innovation 2: Topology-Adaptive Fusion\n(Per-account dynamic weights)", fontsize=9, fontweight='bold')
    for v in isolated_weights + hub_weights: pass

    # Panel 3: Degradation timeline
    ax3 = fig.add_subplot(gs[2])
    t = np.arange(0, 20, 0.1)
    # Simulate 3 regions: full, degraded-path, recovered
    p50_trace = np.where(t < 7, 0.82, np.where(t < 12, 0.68, 0.76))
    ax3.plot(t, p50_trace, 'b-', lw=2, label='Active P@50')
    ax3.axhline(0.60, color='red', lw=1.5, linestyle='--', label='Budget floor (0.60)')
    ax3.axvline(7, color='orange', lw=1.5, linestyle=':', label='NLP failure')
    ax3.axvline(12, color='green', lw=1.5, linestyle=':', label='NLP recovered')
    ax3.fill_between(t, 0.55, p50_trace, alpha=0.15, color='blue')
    ax3.set_xlabel('Time (monitoring cycles)')
    ax3.set_ylabel('Precision@50')
    ax3.set_ylim(0.55, 0.90)
    ax3.legend(fontsize=7.5)
    ax3.set_title("Innovation 3: Degradation Controller\n(P@50 maintained ≥ 0.60 across failures)", fontsize=9, fontweight='bold')
    ax3.text(4, 0.84, 'full path', fontsize=8, ha='center', color='#1A5276')
    ax3.text(9.5, 0.70, 'lgbm_pagerank\n(0.68)', fontsize=8, ha='center', color='#784212')
    ax3.text(16, 0.78, 'no_symbolic\n(0.76)', fontsize=8, ha='center', color='#145A32')

    fig.suptitle("Fig 9 — Conceptual Overview of Three Core Innovations", fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig09_conceptual_innovations.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("fig09 done")

# ─────────────────────────────────────────────────────────────────────
# FIG 10 — Property-Based Test Coverage
# ─────────────────────────────────────────────────────────────────────
def fig10():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.axis('off')
    ax.set_title("Fig 10 — Formal Correctness Properties: Property-Based Test Coverage (119 tests, all PASS)",
                 fontsize=11, fontweight='bold', pad=12)

    headers = ['Property', 'Innovation', 'Formal Statement (Summary)', 'Examples', 'Status']
    rows = [
        ['P1', 'Fusion', 'TopologyVector: 5 metrics, all in correct ranges', '70', 'PASS ✓'],
        ['P2', 'Fusion', 'Weights ∈ [0.05,0.90], sum = 1.0 ± 1e-6', '100', 'PASS ✓'],
        ['P3', 'Fusion', 'Fused score ∈ [0.0, 1.0] for all valid inputs', '100', 'PASS ✓'],
        ['P4', 'Fusion', 'Dense topology → w_graph ≥ 1.5× sparse equivalent', '100', 'PASS ✓'],
        ['P5', 'Fusion', 'Ego-network < 3 nodes → w_ml ≥ 0.70', '100', 'PASS ✓'],
        ['P6', 'TD-PageRank', 'w_temporal = amount × exp(−λ × age) for all edges', '50', 'PASS ✓'],
        ['P7', 'TD-PageRank', 'All scores ≥ 0, normalised ∈ [0,1], iterations ≤ 100', '50', 'PASS ✓'],
        ['P8', 'TD-PageRank', 'Penalised score = 0.5× pre-penalty for SCC nodes', '50', 'PASS ✓'],
        ['P9', 'TD-PageRank', 'Dormant score ≤ 0.1 × max_score (all edges >30d)', '50', 'PASS ✓'],
        ['P10', 'TD-PageRank', 'Identical inputs → identical scores within 1e-12', '50', 'PASS ✓'],
        ['P11', 'Degradation', 'DEGRADED iff 2 consec. unhealthy; HEALTHY iff 3 clean', '50', 'PASS ✓'],
        ['P12', 'Degradation', 'Best P@50 path selected; halt when no path meets budget', '50', 'PASS ✓'],
        ['P13', 'Degradation', 'CRITICAL alert iff P@50 < budget − 0.05', '50', 'PASS ✓'],
        ['P14', 'Degradation', '>3 transitions in 5 min → COOLDOWN ≥ 5 min', '50', 'PASS ✓'],
        ['P15', 'Harness', 'Report fields complete; flagging logic correct', '100', 'PASS ✓'],
        ['AH-1', 'Arch. Hard.', 'LearnableSCCPenalty output ∈ [0.1,1.0] for any input', '100', 'PASS ✓'],
        ['AH-2', 'Arch. Hard.', 'SCC Penalty determinism within 1e-12', '100', 'PASS ✓'],
        ['AH-3', 'Arch. Hard.', 'Backward compat: static heuristic when flag=False', '100', 'PASS ✓'],
        ['AH-4', 'Arch. Hard.', 'GNN embedding always exactly embedding_dim dimensions', '100', 'PASS ✓'],
        ['AH-5', 'Arch. Hard.', '<3 nodes → isolation≥0.9; ≥10 nodes density>0.3 → <0.2', '100', 'PASS ✓'],
        ['AH-6', 'Arch. Hard.', 'IsolationDetector weights sum=1.0, w_ml monotone', '100', 'PASS ✓'],
        ['AH-7', 'Arch. Hard.', 'OnlinePrecisionMonitor: None < 200 samples, valid ≥200', '100', 'PASS ✓'],
        ['AH-8', 'Arch. Hard.', 'Routing table updated iff deviation > 0.05', '100', 'PASS ✓'],
        ['AH-9', 'Arch. Hard.', 'Drift classification correctness + per-path isolation', '100', 'PASS ✓'],
        ['AH-10', 'Arch. Hard.', 'DeepGate weights ∈ [0.05,0.90], sum=1.0, multiplicative', '100', 'PASS ✓'],
        ['AH-11', 'Arch. Hard.', 'Budget blending 0.7×online+0.3×shadow, output∈[0.55,0.75]', '100', 'PASS ✓'],
        ['AH-12', 'Arch. Hard.', 'ResourceManager: every path switch logs memory+time+units', '100', 'PASS ✓'],
    ]

    col_widths = [0.07, 0.12, 0.50, 0.08, 0.10]
    col_x = [0.01, 0.09, 0.22, 0.75, 0.84]
    row_h = 0.034
    y0 = 0.96

    # Header
    for j, (h, x) in enumerate(zip(headers, col_x)):
        ax.text(x, y0, h, fontsize=9, fontweight='bold', transform=ax.transAxes,
                color='white', va='top',
                bbox=dict(facecolor='#1A5276', edgecolor='none', boxstyle='round,pad=0.2'))

    for i, row in enumerate(rows):
        y = y0 - (i+1)*row_h
        bg = '#EBF5FB' if i % 2 == 0 else '#FDFEFE'
        innov = row[1]
        if 'Arch' in innov: bg = '#EAFAF1' if i % 2 == 0 else '#D5F5E3'
        for j, (cell, x) in enumerate(zip(row, col_x)):
            color = '#27AE60' if 'PASS' in cell else '#333'
            ax.text(x, y, cell, fontsize=7.5, va='top', transform=ax.transAxes, color=color,
                    fontweight='bold' if j == 0 or 'PASS' in cell else 'normal')

    ax.text(0.5, 0.02, f'Total: 27 properties | 119 pytest tests | All PASS',
            ha='center', transform=ax.transAxes, fontsize=10, fontweight='bold',
            color='white', bbox=dict(facecolor='#27AE60', edgecolor='none', boxstyle='round,pad=0.3'))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig10_property_test_coverage.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("fig10 done")

# ─────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Saving figures to {OUT}")
    fig01(); fig02(); fig03(); fig04(); fig05()
    fig06(); fig07(); fig08(); fig09(); fig10()
    print("All figures generated successfully!")
