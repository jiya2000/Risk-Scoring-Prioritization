import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import pandas as pd
import numpy as np

def build_hetero_graph(tx_df, kyc_df, features_df):
    """
    Builds a HeteroData object from the dataframes.
    Transactions are edges. Accounts and KYC are nodes.
    """
    data = HeteroData()
    
    # Map account IDs to integer indices
    unique_accounts = features_df['account_id'].unique()
    acc_to_idx = {acc: i for i, acc in enumerate(unique_accounts)}
    
    # Map KYC entities to integer indices
    unique_kyc = kyc_df['name'].unique()
    kyc_to_idx = {name: i for i, name in enumerate(unique_kyc)}
    
    # Node features: Account
    # Excluding account_id and is_laundering
    feat_cols = [c for c in features_df.columns if c not in ['account_id', 'is_laundering', 'account_type']]
    x_acc = torch.tensor(features_df[feat_cols].values, dtype=torch.float)
    data['account'].x = x_acc
    
    if 'is_laundering' in features_df.columns:
        data['account'].y = torch.tensor(features_df['is_laundering'].values, dtype=torch.long)
    
    # Node features: KYC (dummy features for now, e.g., PEP, sanctions)
    kyc_feat_cols = ['pep_flag', 'sanctions_hit']
    # Group by name (entity resolution simplified)
    kyc_nodes = kyc_df.groupby('name')[kyc_feat_cols].max().reset_index()
    x_kyc = torch.tensor(kyc_nodes[kyc_feat_cols].values, dtype=torch.float)
    data['kyc'].x = x_kyc
    
    # Edges: Account -> Account (Transactions)
    # Filter txs that involve known accounts
    tx_df_filtered = tx_df[tx_df['sender_account_id'].isin(acc_to_idx) & tx_df['receiver_account_id'].isin(acc_to_idx)].copy()
    
    src = [acc_to_idx[acc] for acc in tx_df_filtered['sender_account_id']]
    dst = [acc_to_idx[acc] for acc in tx_df_filtered['receiver_account_id']]
    edge_index_tx = torch.tensor([src, dst], dtype=torch.long)
    data['account', 'transacts', 'account'].edge_index = edge_index_tx
    
    # Edge features: Transaction amount, cross border, etc.
    if 'cross_border' in tx_df_filtered.columns:
        edge_attr_tx = torch.tensor(tx_df_filtered[['amount', 'cross_border']].values, dtype=torch.float)
    else:
        edge_attr_tx = torch.tensor(tx_df_filtered[['amount']].values, dtype=torch.float)
    data['account', 'transacts', 'account'].edge_attr = edge_attr_tx
    
    # Temporal Masking prep: store timestamps
    data['account', 'transacts', 'account'].timestamp = torch.tensor(pd.to_datetime(tx_df_filtered['timestamp']).view('int64').values)
    
    # Port Numbering (handling multi-edges)
    # Assign an index to multiple edges between the same src and dst
    # For a real implementation, sort by time and rank.
    # Here we just provide a placeholder to demonstrate the architecture concept.
    port_nums = torch.ones(len(src), dtype=torch.float).unsqueeze(1)
    data['account', 'transacts', 'account'].edge_attr = torch.cat([data['account', 'transacts', 'account'].edge_attr, port_nums], dim=-1)
    
    # Edges: Account -> KYC (Ownership)
    owns_src = [acc_to_idx[row['account_id']] for _, row in kyc_df.iterrows() if row['account_id'] in acc_to_idx]
    owns_dst = [kyc_to_idx[row['name']] for _, row in kyc_df.iterrows() if row['account_id'] in acc_to_idx]
    edge_index_owns = torch.tensor([owns_src, owns_dst], dtype=torch.long)
    data['account', 'owns', 'kyc'].edge_index = edge_index_owns
    
    return data

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # Using SAGEConv as base, will be heterogenized
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class AMLHetGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        self.encoder = to_hetero(GNNEncoder(hidden_channels, out_channels), metadata, aggr='sum')
        self.classifier = torch.nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, ego_id=None):
        """
        ego_id: Optional tensor indicating the target node for ego-centric feature enhancement.
        """
        # If ego_id is provided, we can concatenate a binary indicator to the account features
        # (This is a simplified version of the ego-ID concept from the paper)
        if ego_id is not None:
            batch_size = x_dict['account'].size(0)
            ego_feat = torch.zeros(batch_size, 1, device=x_dict['account'].device)
            ego_feat[ego_id] = 1.0
            x_dict['account'] = torch.cat([x_dict['account'], ego_feat], dim=-1)
            
        z_dict = self.encoder(x_dict, edge_index_dict)
        # We classify accounts
        out = self.classifier(z_dict['account'])
        return out

def train_hetgnn(data, epochs=50, lr=0.01):
    """
    Trains the HetGNN model. Includes temporal masking logic conceptually.
    """
    try:
        model = AMLHetGNN(hidden_channels=64, out_channels=32, metadata=data.metadata())
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Create masks for semi-supervised learning (dummy split)
        num_accounts = data['account'].num_nodes
        train_mask = torch.rand(num_accounts) < 0.8
        test_mask = ~train_mask
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Temporal masking: In a real training loop, we drop edges where timestamp > evaluation_time
            # For simplicity, we just pass the full graph here
            out = model(data.x_dict, data.edge_index_dict)
            loss = criterion(out[train_mask].squeeze(), data['account'].y[train_mask].float())
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")
                
        # Inference
        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
            y_score = torch.sigmoid(out).squeeze().numpy()
            
        return model, y_score, test_mask.numpy()
        
    except Exception as e:
        print(f"ERROR: HetGNN training failed. Falling back to LightGBM. Details: {e}")
        return None, None, None

if __name__ == '__main__':
    from utils.dataloader import load_party_registry, load_ml_features, load_augmented_saml_d
    from models.features import compute_account_features
    
    pr = load_party_registry()
    ml = load_ml_features()
    tx = load_augmented_saml_d()
    features = compute_account_features(ml, pr)
    
    data = build_hetero_graph(tx, pr, features)
    print("Graph built:", data)
    
    model, y_score, test_mask = train_hetgnn(data, epochs=5)
    if model:
        print("HetGNN training completed. Output shape:", y_score.shape)
