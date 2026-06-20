"""
Data Loading Module — Provides clean interfaces for loading all project datasets.

Handles:
- CSV datasets (accounts, transactions, ml_features, graph_edges)
- XML STR reports (flexible schema parsing)
- Graceful error handling for missing files
"""

import os
import pandas as pd
import glob
import xml.etree.ElementTree as ET

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_accounts():
    """Loads the account/KYC registry."""
    path = os.path.join(DATA_DIR, 'accounts.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Ensure dataset is in data/")
    df = pd.read_csv(path)
    return df


def load_transactions():
    """Loads raw transaction data."""
    path = os.path.join(DATA_DIR, 'transactions.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    return df


def load_ml_features():
    """Loads the pre-engineered ML feature matrix."""
    path = os.path.join(DATA_DIR, 'ml_features.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    return df


def load_graph_edges():
    """Loads the sender→receiver edge list for graph construction."""
    path = os.path.join(DATA_DIR, 'graph_edges.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    return df


def load_cached_graph_features():
    """Loads precomputed and regularized graph features."""
    path = os.path.join(DATA_DIR, 'cached_graph_features.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_strs():
    """
    Parses STR XML files from the reports directory.
    Returns a list of dicts (or empty list if reports dir doesn't exist).
    
    Each dict has: str_id, account_number, account_id, reason
    """
    reports_dir = os.path.join(DATA_DIR, 'reports')
    if not os.path.exists(reports_dir):
        # Graceful fallback — return sample STR data for demo purposes
        return _get_sample_strs()

    xml_files = glob.glob(os.path.join(reports_dir, '*.xml'))
    if not xml_files:
        return _get_sample_strs()

    strs = []

    for filepath in xml_files:
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            # Flexible Schema Parsing
            report_id = root.findtext('report_id', default='')
            if not report_id:
                report_id = os.path.basename(filepath)

            reason = root.findtext('reason', default='')

            # Find account number flexibly
            account_number = ''
            account_node = root.find('.//account')
            if account_node is not None and account_node.text:
                account_number = account_node.text
            else:
                # Try account_number tag
                acc_num_node = root.find('.//account_number')
                if acc_num_node is not None and acc_num_node.text:
                    account_number = acc_num_node.text
                else:
                    account_number = 'UNKNOWN'

            # Try to find account_id (numeric)
            account_id = None
            acc_id_node = root.find('.//account_id')
            if acc_id_node is not None and acc_id_node.text:
                try:
                    account_id = int(acc_id_node.text)
                except ValueError:
                    account_id = acc_id_node.text

            strs.append({
                'str_id': report_id,
                'account_number': account_number,
                'account_id': account_id,
                'reason': reason
            })
        except Exception as e:
            print(f"  Warning: Failed to parse {filepath}: {e}")
            continue

    return strs


def _get_sample_strs():
    """
    Returns sample STR narratives for demo/testing when real reports aren't available.
    """
    return [
        {
            'str_id': 'STR-DEMO-001',
            'account_number': 'NP000123',
            'account_id': 5,
            'reason': (
                "Account A5 has received 15 incoming transfers under threshold within 24h, "
                "totalling NPR 2,300,000. Customer could not explain the origin of funds. "
                "The funds were subsequently wired out to an overseas account at Nexus Corp "
                "on 2025-06-14."
            )
        },
        {
            'str_id': 'STR-DEMO-002',
            'account_number': 'NP000456',
            'account_id': 12,
            'reason': (
                "Account #7423 conducted 14 rapid-fire transactions totalling NPR 4.5M "
                "over 48 hours starting on 2025-06-10. All funds originated from 3 shell "
                "companies and were immediately forwarded to 8 different accounts across "
                "multiple jurisdictions. Pattern consistent with layering."
            )
        },
        {
            'str_id': 'STR-DEMO-003',
            'account_number': 'NP000789',
            'account_id': 23,
            'reason': (
                "Previously dormant account NP000789 (no activity for 180 days) suddenly "
                "received NPR 850,000 from an unknown entity on 2025-06-12 and immediately "
                "transferred NPR 840,000 cross-border. Customer profile does not match "
                "transaction pattern."
            )
        }
    ]


if __name__ == '__main__':
    # Test loaders
    print("Testing data loaders...")
    try:
        acc = load_accounts()
        print(f"  Accounts: {acc.shape}")
    except FileNotFoundError as e:
        print(f"  {e}")

    try:
        ml = load_ml_features()
        print(f"  ML Features: {ml.shape}")
    except FileNotFoundError as e:
        print(f"  {e}")

    try:
        ge = load_graph_edges()
        print(f"  Graph Edges: {ge.shape}")
    except FileNotFoundError as e:
        print(f"  {e}")

    strs = load_strs()
    print(f"  STRs loaded: {len(strs)}")
    if strs:
        print(f"  Sample: {strs[0]['reason'][:60]}...")
