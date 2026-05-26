import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGCNLayer(nn.Module):
    """
    A single Graph Convolutional Layer designed for dense adjacency matrices.
    Features are propagated as out = Adj * X * W.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.15):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (Batch, ROIs, In_Features)
        # adj: (Batch, ROIs, ROIs)
        support = self.linear(x)
        out = torch.bmm(adj, support)
        return self.dropout(self.act(out))


class fMRIGCN(nn.Module):
    """
    Spatio-Temporal GCN for resting-state fMRI classification.
    Uses functional connectivity as graph edges and BOLD signals as node features.
    """
    def __init__(self, n_rois: int, n_time: int = 60, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.n_rois = n_rois
        
        # Temporal feature embedding per ROI
        self.temporal_mlp = nn.Sequential(
            nn.Linear(n_time, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Graph convolution layers
        self.gcn1 = DenseGCNLayer(hidden_dim, hidden_dim, dropout)
        self.gcn2 = DenseGCNLayer(hidden_dim, hidden_dim, dropout)
        
        # Readout and classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, activation: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        # activation: (B, 1, Time, ROIs) -> (B, ROIs, Time)
        x = activation.squeeze(1).permute(0, 2, 1)
        
        # 1. Local Node Feature Normalization (Z-score along time dimension)
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True) + 1e-5
        x = (x - x_mean) / x_std
        
        # 2. Graph Adjacency Matrix Preprocessing
        adj = corr.squeeze(1)
        adj = F.relu(adj)  # Retain positive synchronous correlations
        adj = torch.where(adj > 0.15, adj, torch.zeros_like(adj))  # Sparsification threshold
        
        # Add self-loops to preserve central node identities
        eye = torch.eye(self.n_rois, device=adj.device).unsqueeze(0).expand_as(adj)
        adj = adj + eye
        
        # Symmetric Degree Normalization: D^(-1/2) * A * D^(-1/2)
        degree = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
        d_mat = torch.diag_embed(d_inv_sqrt)
        adj_norm = torch.bmm(torch.bmm(d_mat, adj), d_mat)
        
        # 3. Message Passing & Feature Propagation
        x_base = self.temporal_mlp(x)
        x_gcn = self.gcn1(x_base, adj_norm)
        x_gcn = self.gcn2(x_gcn, adj_norm)
        
        # Residual skip connection to prevent signal over-smoothing
        x_out = x_base + x_gcn
        
        # 4. Global Average Pooling over ROIs and Classification
        x_pooled = x_out.mean(dim=1)
        return self.classifier(x_pooled)