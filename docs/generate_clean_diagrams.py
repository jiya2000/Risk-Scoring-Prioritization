# -*- coding: utf-8 -*-
"""
Generate clean, Mermaid-style flowchart PNGs for:
  fig01  — System Architecture  (top-down flowchart)
  fig02  — TD-PageRank Flowchart (top-down flowchart)
Saves to docs/artifacts/
"""
import os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(OUT, exist_ok=True)
DPI = 180

# ─── palette (Mermaid default-ish) ───────────────────────────────────
C = dict(
    blue  ='#4A90D9',   # process / step
    green ='#27AE60',   # start / end
    yellow='#F39C12',   # decision / loop
    orange='#E67E22',   # model / gate
    red   ='#E74C3C',   # penalty / critical
    purple='#8E44AD',   # rules
    teal  ='#16A085',   # controller
    grey  ='#95A5A6',   # aggregation
    lime  ='#2ECC71',   # hardening
    peach ='#FAD7A0',   # dashboard
    lblue ='#AED6F1',   # data / NLP
    white ='#FDFEFE',
    dark  ='#2C3E50',
)

def fig(w, h): return plt.subplots(figsize=(w, h))

def box(ax, cx, cy, w, h, text, fill, fs=9.5, radius=0.04, ec='#2C3E50', lw=1.5):
    """Draw a rounded rectangle with centred text."""
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad={radius}",
                       facecolor=fill, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='white', multialignment='center',
            linespacing=1.45, zorder=4,
            path_effects=[pe.withStroke(linewidth=0, foreground='none')])

def diamond(ax, cx, cy, hw, hh, text, fill=C['yellow'], fs=9):
    """Draw a diamond (decision) shape."""
    xs = [cx, cx+hw, cx, cx-hw, cx]
    ys = [cy+hh, cy, cy-hh, cy, cy+hh]
    ax.fill(xs, ys, color=fill, ec=C['dark'], lw=1.5, zorder=3)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='white', multialignment='center',
            linespacing=1.3, zorder=4)

def arr(ax, x1, y1, x2, y2, label='', lc='#2C3E50', lw=1.8, curved=False):
    """Draw an arrow between two points."""
    if curved:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=lc, lw=lw,
                                    connectionstyle='arc3,rad=0.35'), zorder=2)
    else:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=lc, lw=lw), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.06, my+0.06, label, fontsize=7.5,
                color='#555', style='italic', zorder=5)

def label(ax, x, y, text, fs=8, color='#333', bold=False):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            color=color, fontweight='bold' if bold else 'normal')

# ══════════════════════════════════════════════════════════════════════
# FIG 02  — TD-PageRank Algorithm Flowchart  (clean Mermaid-style)
# ══════════════════════════════════════════════════════════════════════
def make_fig02():
    W, H = 7, 17
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    # ── node dimensions ──────────────────────────────────────────────
    BW = 4.6   # box width
    BH = 0.72  # box height (normal)
    BH2= 0.85  # box height (taller)
    CX = W/2   # center x

    # ── y positions  (top → bottom, step ~1.5) ───────────────────────
    ys = [16.0, 14.4, 12.8, 11.2, 9.6, 8.0, 6.4, 4.6, 3.0, 1.3]
    labels = [
        ("START", C['green'], BH, "● START"),
        ("load",  C['blue'],  BH, "Load edge list\n(sender, receiver, amount, date)"),
        ("wt",    C['blue'],  BH, "Compute temporal weights\nw = amount × exp(−λ × EdgeAge)"),
        ("burst", C['blue'],  BH, "Detect burst-velocity senders\nburst_mult = 1 + (window_cnt / total_cnt)"),
        ("graph", C['blue'],  BH, "Build directed graph G\nwith  w_temporal × burst_mult  on each edge"),
        ("init",  C['blue'],  BH, "Initialise  r[v] = 1/N  for all nodes v"),
        ("iter",  C['orange'],BH, "Power iteration step\nr_new = (1−d)/N + d × Mᵀ · r"),
        ("conv",  C['yellow'],BH*1.2, None),   # decision — drawn separately
        ("scc",   C['orange'],BH2, "Detect SCCs of size > 2\nApply directional SCC penalty:\nCollector 0.25×  |  Distributor 0.50×  |  Balanced 0.375×"),
        ("dorm",  C['red'],   BH,  "Detect dormant nodes  (all edges > 30 days)\nCap score ≤ 0.1 × max_score"),
    ]

    # ── draw boxes (skip decision) ────────────────────────────────────
    bh_list = [BH, BH, BH, BH, BH, BH, BH, None, BH2, BH]
    for i, (y, bh) in enumerate(zip(ys, bh_list)):
        if bh is None: continue
        _, fill, _, txt = labels[i]
        box(ax, CX, y, BW, bh, txt, fill, fs=9.5 if i==0 else 9)

    # ── decision diamond ─────────────────────────────────────────────
    dy = ys[7]
    diamond(ax, CX, dy, 2.1, 0.55,
            "‖ r_new − r ‖₁ < 1e-6\n    or  iter ≥ 100 ?",
            fill=C['yellow'], fs=9)

    # ── final normalise box ───────────────────────────────────────────
    box(ax, CX, ys[-1], BW, BH,
        "Min-max normalise  →  [0, 1]\nReturn  TDPageRankResult", C['green'], fs=9.5)

    # ── straight arrows down ──────────────────────────────────────────
    for i in range(len(ys)-1):
        if i == 6: continue    # skip iter→decision (handled below)
        if i == 7: continue    # skip decision→scc   (handled below)
        y_from = ys[i] - (bh_list[i] or BH)/2 - 0.02
        y_to   = ys[i+1] + (bh_list[i+1] or 0.55)/2 + 0.02
        arr(ax, CX, y_from, CX, y_to)

    # iter → decision
    arr(ax, CX, ys[6]-BH/2-0.02, CX, dy+0.55+0.02)
    # decision → YES → scc
    arr(ax, CX, dy-0.55-0.02, CX, ys[8]+BH2/2+0.02)
    ax.text(CX+0.18, (dy-0.55+ys[8]+BH2/2)/2, "YES",
            fontsize=9, color=C['green'], fontweight='bold')
    # decision → NO  → loop back to iter  (clean right-angle path)
    # Path: diamond right tip → go right → go up → arrow into iter box right edge
    no_x_out  = CX + 2.1          # diamond right tip x
    no_y      = dy                  # diamond center y
    loop_x    = CX + BW/2 + 0.9   # right margin for the loop line
    iter_y    = ys[6]              # iteration box center y
    iter_r    = CX + BW/2          # iteration box right edge x

    # Horizontal line from diamond right tip to loop margin
    ax.plot([no_x_out, loop_x], [no_y, no_y], color=C['dark'], lw=1.8, zorder=2)
    # Vertical line going up from diamond level to iteration level
    ax.plot([loop_x, loop_x], [no_y, iter_y], color=C['dark'], lw=1.8, zorder=2)
    # Arrow from loop margin into the iteration box right edge
    ax.annotate('', xy=(iter_r+0.02, iter_y), xytext=(loop_x, iter_y),
                arrowprops=dict(arrowstyle='->', color=C['dark'], lw=1.8), zorder=2)
    # "NO" label
    ax.text(loop_x + 0.2, (no_y + iter_y) / 2, "NO",
            fontsize=10, color=C['red'], fontweight='bold', va='center')

    # scc → dorm
    arr(ax, CX, ys[8]-BH2/2-0.02, CX, ys[9]+BH/2+0.02)
    # dorm → normalise
    arr(ax, CX, ys[9]-BH/2-0.02, CX, ys[-1]+BH/2+0.02)

    # ── title ─────────────────────────────────────────────────────────
    ax.set_title("Fig 2 — TD-PageRank Algorithm Flowchart",
                 fontsize=13, fontweight='bold', pad=10, color=C['dark'])

    plt.tight_layout(pad=0.4)
    out = os.path.join(OUT, 'fig02_tdpagerank_flowchart.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(fig)
    print(f"fig02 saved  ({os.path.getsize(out)//1024} KB)")

# ══════════════════════════════════════════════════════════════════════
# FIG 01  — Full System Architecture  (clean Mermaid-style top-down)
# ══════════════════════════════════════════════════════════════════════
def make_fig01():
    W, H = 14, 18
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    # ── layout constants ─────────────────────────────────────────────
    # Row y-centres (top to bottom)
    RY = dict(
        data  =17.0,
        feat  =15.1,
        core  =13.1,
        fusion=11.1,
        ctrl  =9.0,
        hard  =7.0,
        out   =4.8,
    )
    BH  = 1.1    # standard box height
    BH2 = 1.25   # taller box height

    # ── Row A: Data inputs (full width, one box) ──────────────────────
    box(ax, W/2, RY['data'], 5.5, BH,
        "Input Data\n"
        "transactions.csv  |  accounts.csv  |  graph_edges.csv",
        C['lblue'], fs=10, ec='#2980B9')

    # ── Row B: Feature Engineering + LightGBM ────────────────────────
    box(ax, 3.2, RY['feat'], 5.0, BH,
        "Feature Engineering\n40+ features  (temporal, KYC, graph)",
        C['green'], fs=10)
    box(ax, 9.8, RY['feat'], 4.5, BH,
        "LightGBM Classifier\n→  S_ml  ∈ [0, 1]",
        C['green'], fs=10)

    # ── Row C: TD-PageRank | Ego-Network | TopologyAttentionGate ─────
    box(ax, 2.3, RY['core'], 3.8, BH2,
        "TD-PageRank Engine\n"
        "w = amt × exp(−λ×age) × burst\n"
        "Directional SCC Penalty\n→  S_graph",
        C['yellow'], fs=9.5)
    box(ax, 7.0, RY['core'], 3.8, BH2,
        "Ego-Network Extractor\n"
        "2-hop subgraph\n"
        "→  TopologyVector  (6 metrics)",
        C['yellow'], fs=9.5)
    box(ax, 11.8, RY['core'], 3.8, BH2,
        "TopologyAttentionGate\n"
        "multiplicative gating\n"
        "→  [w_ml, w_graph, w_rules]",
        C['orange'], fs=9.5)

    # ── Row D: Adaptive Fusion + Symbolic Rules ───────────────────────
    box(ax, 4.0, RY['fusion'], 6.5, BH,
        "Adaptive Fusion Engine\n"
        "fused = w_ml·S_ml  +  w_graph·S_graph  +  w_rules·S_rules",
        C['orange'], fs=10)
    box(ax, 11.2, RY['fusion'], 4.6, BH,
        "Symbolic Rule Engine\n"
        "10 FATF Typologies  →  S_rules",
        C['purple'], fs=10)

    # ── Row E: Degradation Controller + Account Aggregation ──────────
    box(ax, 4.0, RY['ctrl'], 6.5, BH2,
        "Degradation Controller  (Innovation 3)\n"
        "8 Execution Paths  |  P@50 ≥ 0.60 guaranteed\n"
        "PrecisionDriftDetector  +  AdaptivePrecisionBudget",
        C['teal'], fs=9.5)
    box(ax, 11.2, RY['ctrl'], 4.6, BH2,
        "Account Risk Aggregation\n"
        "time-decay + burst + high-risk ratio\n"
        "→  Priority Queue  (Critical/High/Medium/Low)",
        C['grey'], fs=9.5)

    # ── Row F: Architecture Hardening Strip ──────────────────────────
    box(ax, W/2, RY['hard'], W-0.6, BH2,
        "Architecture Hardening Layer\n"
        "(all ArchitectureHardeningConfig flags default = False  →  fully backward-compatible)\n"
        "LearnableSCCPenalty  |  TopologyEmbeddingNetwork (GNN)  |  IsolationDetector\n"
        "DeepTopologyAttentionGate  |  OnlinePrecisionMonitor  |  EnhancedConceptDriftDetector  |  ResourceManager (§101)",
        C['lime'], fs=9, ec='#1E8449')

    # ── Row G: NLP + Dashboard ────────────────────────────────────────
    box(ax, 3.2, RY['out'], 5.2, BH,
        "NLP Pipeline\n"
        "spaCy NER  +  SmolLM-135M\n"
        "STR Narrative Generation",
        C['lblue'], fs=9.5, ec='#2980B9')
    box(ax, 10.5, RY['out'], 6.8, BH,
        "Streamlit Analyst Dashboard\n"
        "Priority Queue  |  Case Investigation  |  Network Viz  |  SHAP  |  Export",
        C['peach'], fs=10, ec='#E67E22')

    # ── Arrows ────────────────────────────────────────────────────────
    # Data → Feature Eng
    arr(ax, 5.0, RY['data']-BH/2, 3.2, RY['feat']+BH/2)
    # Data → TD-PageRank  (left branch)
    arr(ax, 3.8, RY['data']-BH/2, 2.3, RY['core']+BH2/2)
    # Feature Eng → LightGBM
    arr(ax, 5.7, RY['feat'], 7.55, RY['feat'], 'features')
    # LightGBM → TopologyAttentionGate
    arr(ax, 11.8, RY['feat']-BH/2, 11.8, RY['core']+BH2/2, 'S_ml')
    # TD-PageRank → Ego-Network
    arr(ax, 4.2, RY['core'], 5.1, RY['core'], '')
    # Ego-Network → TopologyAttentionGate
    arr(ax, 8.9, RY['core'], 9.9, RY['core'], 'topology\nvector')
    # TopologyAttentionGate → Adaptive Fusion  (diagonal)
    arr(ax, 11.8, RY['core']-BH2/2, 6.0, RY['fusion']+BH/2, 'weights')
    # TD-PageRank → Adaptive Fusion  (straight down)
    arr(ax, 2.3, RY['core']-BH2/2, 2.3, RY['fusion']+BH/2, 'S_graph')
    # Symbolic Rules → Adaptive Fusion
    arr(ax, 8.95, RY['fusion'], 7.25, RY['fusion'], 'S_rules')
    # Adaptive Fusion → Degradation Controller
    arr(ax, 4.0, RY['fusion']-BH/2, 4.0, RY['ctrl']+BH2/2)
    # Degradation Controller → Account Aggregation
    arr(ax, 7.25, RY['ctrl'], 8.95, RY['ctrl'])
    # Account Aggregation → Dashboard
    arr(ax, 11.2, RY['ctrl']-BH2/2, 10.5, RY['out']+BH/2)
    # Degradation Controller → NLP
    arr(ax, 2.5, RY['ctrl']-BH2/2, 2.5, RY['out']+BH/2)
    # Hardening ↔ core components (dashed style indicator)
    for xp in [2.3, 7.0, 11.8]:
        ax.annotate('', xy=(xp, RY['hard']+BH2/2),
                    xytext=(xp, RY['ctrl']-BH2/2-0.2),
                    arrowprops=dict(arrowstyle='->', color='#1E8449',
                                    lw=1.2, linestyle='dashed'))

    # ── title ─────────────────────────────────────────────────────────
    ax.set_title("Fig 1 — AML Risk Scoring Platform: Full System Architecture",
                 fontsize=14, fontweight='bold', pad=14, color=C['dark'])

    plt.tight_layout(pad=0.5)
    out = os.path.join(OUT, 'fig01_system_architecture.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(fig)
    print(f"fig01 saved  ({os.path.getsize(out)//1024} KB)")


if __name__ == '__main__':
    print(f"Saving to: {OUT}")
    make_fig02()
    make_fig01()
    print("Done.")
