"""
Integration tests for training convergence of learnable components.

Validates:
- Learnable SCC penalty achieves MAD >= 0.01 from standard PageRank
- GNN topology achieves P@50 >= 0.82 on temporal test set
- Deep gate achieves >= 10% relative P@50 improvement over static baseline
"""
