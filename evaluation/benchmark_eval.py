"""
Elliptic Benchmark Evaluator — TD-PageRank vs Standard PageRank.

Compares TD-PageRank scores against standard NetworkX PageRank on the
Elliptic Bitcoin transaction dataset (or synthetic mock data when the
dataset is absent). Produces structured benchmark results demonstrating
the differentiation between TD-PageRank and standard PageRank.

Satisfies Requirements 6.1, 6.2, 6.3, 6.4:
  - 6.1: load_elliptic_dataset accepts a path and returns a graph DataFrame
  - 6.2: Returns mock results with WARNING if dataset file is absent
  - 6.3: Asserts mean absolute difference >= 0.01 between algorithms
  - 6.4: Resides in evaluation/benchmark_eval.py as a standalone module
"""

import sys
import os
import logging
from datetime import date, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import networkx as nx

# ── Project root on path ─────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.td_pagerank import TDPageRankEngine

logger = logging.getLogger(__name__)


class EllipticBenchmarkEvaluator:
    """
    Benchmark evaluator comparing TD-PageRank against standard PageRank
    on the Elliptic Bitcoin transaction dataset.

    When the Elliptic dataset is not available locally, the evaluator
    gracefully degrades to synthetic mock data with clearly labelled
    provenance (is_mock=True in results).
    """

    def load_elliptic_dataset(self, path: str) -> pd.DataFrame:
        """
        Load the Elliptic dataset from the given filesystem path.

        Args:
            path: Filesystem path to the Elliptic dataset CSV file.
                  Expected columns: Sender_account, Receiver_account,
                  amount_local_npr, Date.

        Returns:
            pd.DataFrame with graph edge data. If the file does not exist,
            returns a synthetic mock DataFrame with temporal variation in
            edge ages and logs a WARNING.
        """
        try:
            df = pd.read_csv(path)
            logger.info("Loaded Elliptic dataset from %s (%d edges)", path, len(df))
            return df
        except FileNotFoundError:
            logger.warning(
                "Elliptic dataset not found at '%s'. "
                "Returning synthetic mock data for benchmark evaluation.",
                path,
            )
            return self._generate_mock_graph_data()

    def run_td_pagerank_vs_standard(
        self, graph_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute both TD-PageRank and standard PageRank, then compare.

        Asserts that the mean absolute difference between the two algorithms
        is at least 0.01, demonstrating that TD-PageRank produces materially
        different (and temporally-aware) scores compared to standard PageRank.

        Args:
            graph_df: DataFrame with columns [Sender_account, Receiver_account,
                      amount_local_npr, Date].

        Returns:
            Dictionary with keys:
              - td_pagerank_scores: Dict[str, float]
              - standard_pagerank_scores: Dict[str, float]
              - mean_absolute_difference: float
              - node_count: int
              - is_mock: bool (True if mock data was used)
        """
        # Determine if this is mock data by checking for the _is_mock column
        is_mock = "_is_mock" in graph_df.columns

        # --- Compute TD-PageRank scores ---
        engine = TDPageRankEngine(
            half_life_days=7.0,
            damping=0.85,
            burst_window_days=3,
        )
        result = engine.compute(graph_df)
        td_scores = result.scores

        # --- Compute Standard PageRank scores ---
        G = nx.DiGraph()
        for _, row in graph_df.iterrows():
            sender = str(row["Sender_account"])
            receiver = str(row["Receiver_account"])
            amount = float(row["amount_local_npr"])
            # Standard PageRank uses raw amount as weight (no temporal decay)
            if G.has_edge(sender, receiver):
                G[sender][receiver]["weight"] += amount
            else:
                G.add_edge(sender, receiver, weight=amount)

        standard_scores = nx.pagerank(G, alpha=0.85, weight="weight")

        # --- Compute mean absolute difference ---
        all_nodes = set(td_scores.keys()) | set(standard_scores.keys())
        node_count = len(all_nodes)

        differences = []
        for node in all_nodes:
            td_val = td_scores.get(node, 0.0)
            std_val = standard_scores.get(node, 0.0)
            differences.append(abs(td_val - std_val))

        mean_abs_diff = float(np.mean(differences)) if differences else 0.0

        # Assert the differentiation threshold
        assert mean_abs_diff >= 0.01, (
            f"TD-PageRank vs Standard PageRank mean absolute difference "
            f"({mean_abs_diff:.6f}) is below the 0.01 threshold. "
            f"The temporal decay mechanism must produce materially different scores."
        )

        return {
            "td_pagerank_scores": td_scores,
            "standard_pagerank_scores": standard_scores,
            "mean_absolute_difference": mean_abs_diff,
            "node_count": node_count,
            "is_mock": is_mock,
        }

    def _generate_mock_graph_data(self) -> pd.DataFrame:
        """
        Generate synthetic mock graph data with temporal variation in edge ages.

        Creates a small transaction network with:
        - A mix of recent edges (0-3 days old) and old edges (30-90 days old)
        - Multiple senders and receivers to ensure graph connectivity
        - Enough variation that TD-PageRank and standard PageRank produce
          materially different scores (mean abs diff >= 0.01)

        Returns:
            pd.DataFrame with columns: Sender_account, Receiver_account,
            amount_local_npr, Date, _is_mock
        """
        reference_date = date.today()

        # Recent edges (high temporal weight in TD-PageRank)
        recent_edges = [
            ("ACC_001", "ACC_002", 50000.0, reference_date - timedelta(days=1)),
            ("ACC_001", "ACC_003", 75000.0, reference_date - timedelta(days=0)),
            ("ACC_002", "ACC_004", 30000.0, reference_date - timedelta(days=2)),
            ("ACC_003", "ACC_005", 60000.0, reference_date - timedelta(days=1)),
            ("ACC_004", "ACC_005", 40000.0, reference_date - timedelta(days=3)),
            ("ACC_005", "ACC_001", 25000.0, reference_date - timedelta(days=2)),
        ]

        # Old edges (low temporal weight in TD-PageRank, same raw weight in standard)
        old_edges = [
            ("ACC_006", "ACC_001", 80000.0, reference_date - timedelta(days=45)),
            ("ACC_007", "ACC_002", 90000.0, reference_date - timedelta(days=60)),
            ("ACC_006", "ACC_003", 70000.0, reference_date - timedelta(days=50)),
            ("ACC_008", "ACC_004", 55000.0, reference_date - timedelta(days=75)),
            ("ACC_007", "ACC_005", 65000.0, reference_date - timedelta(days=90)),
            ("ACC_008", "ACC_006", 45000.0, reference_date - timedelta(days=35)),
            ("ACC_009", "ACC_007", 85000.0, reference_date - timedelta(days=80)),
            ("ACC_009", "ACC_008", 35000.0, reference_date - timedelta(days=55)),
        ]

        all_edges = recent_edges + old_edges

        df = pd.DataFrame(
            all_edges,
            columns=["Sender_account", "Receiver_account", "amount_local_npr", "Date"],
        )
        # Convert Date to string format matching project conventions
        df["Date"] = df["Date"].apply(lambda d: d.strftime("%Y-%m-%d"))
        # Mark as mock data
        df["_is_mock"] = True

        return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    evaluator = EllipticBenchmarkEvaluator()

    # Attempt to load Elliptic dataset (will fall back to mock)
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "elliptic_txs_edgelist.csv",
    )
    graph_df = evaluator.load_elliptic_dataset(dataset_path)

    # Run benchmark comparison
    results = evaluator.run_td_pagerank_vs_standard(graph_df)

    print("\n=== Elliptic Benchmark Results ===")
    print(f"  Node count:                {results['node_count']}")
    print(f"  Mean absolute difference:  {results['mean_absolute_difference']:.6f}")
    print(f"  Is mock data:              {results['is_mock']}")
    print(f"  Threshold (>= 0.01):       PASS")
