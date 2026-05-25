"""
Modelos para rs-fMRI:
- fMRIGCN: Graph Convolutional Network. Usa la matriz de correlación como 
  grafo (adyacencia) y las activaciones temporales como características de nodo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fmri_dataset import CHUNK_LENGTH as N_TIME

def _upper_triangle_vector(corr: torch.Tensor) -> torch.Tensor:
    """Extrae el triángulo superior de la matriz de correlación."""
    _, _, r, _ = corr.shape
    idx = torch.triu_indices(r, r, offset=1, device=corr.device)
    return corr[:, 0, idx[0], idx[1]]

class DenseGCNLayer(nn.Module):
    """Capa GCN para grafos densos (como matrices de correlación fMRI)."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (Batch, ROIs, Features)
        # adj: (Batch, ROIs, ROIs)
        
        # Transformación de características
        support = self.linear(x) 
        
        # Multiplicación por la matriz de adyacencia (Graph convolution)
        # A * X * W
        out = torch.bmm(adj, support)
        return self.dropout(self.act(out))

class fMRIGCN(nn.Module):
    """
    GCN que integra series temporales y topología de red.
    """
    def __init__(self, n_rois: int, n_time: int = N_TIME, hidden_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.n_rois = n_rois
        
        self.temporal_mlp = nn.Sequential(
            nn.Linear(n_time, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Capas GCN
        self.gcn1 = DenseGCNLayer(hidden_dim, hidden_dim, dropout)
        self.gcn2 = DenseGCNLayer(hidden_dim, hidden_dim, dropout)
        
        # Clasificador final (Global Average Pooling sobre los nodos)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2) # 2 clases: Jóvenes vs Adultos
        )

    def forward(self, activation: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        # activation: (B, 1, Time, ROIs) -> Convertir a (B, ROIs, Time)
        x = activation.squeeze(1).permute(0, 2, 1)
        
        # corr: (B, 1, ROIs, ROIs) -> Convertir a (B, ROIs, ROIs)
        adj = corr.squeeze(1)
        
        # Normalizar adyacencia (opcional pero recomendado: usar valor absoluto)
        # En fMRI, las anticorrelaciones también contienen info, pero a nivel de grafo
        # los pesos negativos pueden desestabilizar. Pasamos a absoluto y sumamos la identidad.
        adj = torch.abs(adj)
        eye = torch.eye(self.n_rois, device=adj.device).unsqueeze(0).expand_as(adj)
        adj = adj + eye # Self-loops asegurados
        
        # Extraer características temporales
        x = self.temporal_mlp(x) # (B, ROIs, Hidden)
        
        # Pasar por las GCN
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        
        # Pooling global sobre los ROIs (Nodos)
        x_pooled = x.mean(dim=1) # (B, Hidden)
        
        return self.classifier(x_pooled)