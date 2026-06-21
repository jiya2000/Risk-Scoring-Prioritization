import networkx as nx
import pandas as pd
import numpy as np
import sys
import os

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataloader import load_graph_edges
from models.td_pagerank import TDPageRankEngine


def extract_graph_features(edges_df, use_td_pagerank: bool = True):
    """
    Constructs a directed graph and calculates comprehensive centrality and
    structural metrics for all nodes (accounts).

    Args:
        edges_df: DataFrame with columns [Sender_account, Receiver_account,
                  amount_local_npr]. When use_td_pagerank=True, a Date column
                  is also required for temporal decay computation.
        use_td_pagerank: When True (default), uses TDPageRankEngine instead of
                         nx.pagerank(). Stores normalized scores as gf_pagerank
                         (replacing the nx.pagerank value) and additionally
                         stores gf_td_pagerank_score, gf_cycle_member, and
                         gf_decay_impact. Falls back to nx.pagerank() if the
                         Date column is absent or TD-PageRank raises an error.

    Features extracted:
    - In/Out Degree (unweighted and weighted)
    - PageRank (weighted) — TD-PageRank or nx.pagerank depending on flag
    - gf_td_pagerank_score  (TD-PageRank only)
    - gf_cycle_member       (TD-PageRank only)
    - gf_decay_impact       (TD-PageRank only)
    - Clustering Coefficient
    - Betweenness Centrality (identifies bridge/intermediary nodes)
    - HITS Hub & Authority scores (identifies distributors vs collectors)
    - Degree Ratio (in_degree / total_degree) — proxy for sink vs source behavior
    - Transaction Reciprocity (fraction of counterparties with bilateral flow)
    - Unique Counterparties (distinct in/out neighbors)
    - Cycle Participation (whether node belongs to a directed cycle)
    """
    print("Building Directed Graph...")
    G = nx.from_pandas_edgelist(
        edges_df,
        source='Sender_account',
        target='Receiver_account',
        edge_attr='amount_local_npr',
        create_using=nx.DiGraph()
    )

    print(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # === Core Degree Features ===
    print("Calculating degree metrics...")
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    in_degree_weight = dict(G.in_degree(weight='amount_local_npr'))
    out_degree_weight = dict(G.out_degree(weight='amount_local_npr'))

    # === PageRank ===
    print("Calculating PageRank...")
    # Attempt TD-PageRank if requested and Date column is present
    td_pagerank_result = None
    if use_td_pagerank and 'Date' in edges_df.columns:
        try:
            # Use max(Date) as reference_date for temporal decay
            reference_date = pd.to_datetime(edges_df['Date']).max()
            if not pd.isna(reference_date):
                reference_date = reference_date.date() if hasattr(reference_date, 'date') else reference_date
                engine = TDPageRankEngine()
                td_pagerank_result = engine.compute(edges_df, reference_date=reference_date)
                print("  Using TD-PageRank (temporal-decay) for gf_pagerank.")
            else:
                print("  TD-PageRank: Date column has all-NaT values; falling back to nx.pagerank().")
        except Exception as e:
            print(f"  TD-PageRank failed ({e}); falling back to nx.pagerank().")
            td_pagerank_result = None
    elif use_td_pagerank:
        print("  TD-PageRank requested but no Date column found; falling back to nx.pagerank().")

    if td_pagerank_result is None:
        pagerank = nx.pagerank(G, weight='amount_local_npr', alpha=0.85)

    # === Clustering Coefficient (undirected) ===
    print("Calculating Clustering Coefficient...")
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)

    # === Betweenness Centrality (sampled for performance) ===
    print("Calculating Betweenness Centrality (sampled k=500)...")
    n_nodes = G.number_of_nodes()
    k_sample = min(500, n_nodes)
    betweenness = nx.betweenness_centrality(G, k=k_sample, weight='amount_local_npr', normalized=True)

    # === HITS (Hub & Authority) ===
    print("Calculating HITS Hub & Authority scores...")
    try:
        hubs, authorities = nx.hits(G, max_iter=100, normalized=True)
    except nx.PowerIterationFailedConvergence:
        # Fallback: assign uniform scores
        hubs = {n: 1.0 / n_nodes for n in G.nodes()}
        authorities = {n: 1.0 / n_nodes for n in G.nodes()}

    # === Derived Structural Features ===
    print("Computing derived structural features...")

    # Precompute predecessors and successors sets for reciprocity
    predecessors = {n: set(G.predecessors(n)) for n in G.nodes()}
    successors = {n: set(G.successors(n)) for n in G.nodes()}

    # === Cycle Participation (per-node) ===
    print("Detecting cycle participation...")
    # Find all nodes that are part of any strongly connected component with size > 1
    sccs = list(nx.strongly_connected_components(G))
    nodes_in_cycles = set()
    for scc in sccs:
        if len(scc) > 1:
            nodes_in_cycles.update(scc)

    # === Assemble Features ===
    print("Aggregating graph features...")
    features = []
    for node in G.nodes():
        in_d = in_degree.get(node, 0)
        out_d = out_degree.get(node, 0)
        total_d = in_d + out_d

        # Degree ratio: fraction of connections that are incoming (sink-like behavior)
        degree_ratio = in_d / total_d if total_d > 0 else 0.5

        # Flow ratio: weighted_in / (weighted_in + weighted_out)
        w_in = in_degree_weight.get(node, 0)
        w_out = out_degree_weight.get(node, 0)
        flow_ratio = w_in / (w_in + w_out) if (w_in + w_out) > 0 else 0.5

        # Unique counterparties
        unique_in_neighbors = len(predecessors[node])
        unique_out_neighbors = len(successors[node])

        # Transaction reciprocity: fraction of counterparties with bilateral flow
        all_neighbors = predecessors[node].union(successors[node])
        if len(all_neighbors) > 0:
            bilateral = predecessors[node].intersection(successors[node])
            reciprocity = len(bilateral) / len(all_neighbors)
        else:
            reciprocity = 0.0

        # Cycle membership
        is_in_cycle = 1 if node in nodes_in_cycles else 0

        row = {
            'account_id': node,
            'gf_in_degree': in_d,
            'gf_out_degree': out_d,
            'gf_weighted_in_degree': w_in,
            'gf_weighted_out_degree': w_out,
            'gf_clustering_coefficient': clustering.get(node, 0),
            'gf_betweenness': betweenness.get(node, 0),
            'gf_hub_score': hubs.get(node, 0),
            'gf_authority_score': authorities.get(node, 0),
            'gf_degree_ratio': degree_ratio,
            'gf_flow_ratio': flow_ratio,
            'gf_unique_in_neighbors': unique_in_neighbors,
            'gf_unique_out_neighbors': unique_out_neighbors,
            'gf_reciprocity': reciprocity,
            'gf_in_cycle': is_in_cycle,
        }

        # PageRank — TD-PageRank path or nx.pagerank fallback
        node_str = str(node)
        if td_pagerank_result is not None:
            # Normalized score replaces gf_pagerank (so downstream models see same column)
            norm_score = td_pagerank_result.normalized_scores.get(node_str, 0.0)
            row['gf_pagerank'] = norm_score
            row['gf_td_pagerank_score'] = norm_score
            row['gf_cycle_member'] = int(td_pagerank_result.cycle_member.get(node_str, False))
            row['gf_decay_impact'] = td_pagerank_result.decay_impact.get(node_str, 0.0)
        else:
            row['gf_pagerank'] = pagerank.get(node, 0)

        features.append(row)

    df_features = pd.DataFrame(features)
    print(f"Graph feature extraction complete. {len(df_features)} nodes, {len(df_features.columns)-1} features.")
    return df_features


if __name__ == '__main__':
    edges = load_graph_edges()
    gf = extract_graph_features(edges)
    print(gf.head())
    print("\nFeature statistics:")
    print(gf.describe())
