import time
import argparse
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.utils import degree, contains_self_loops
from torch_geometric.data import Data
from torch_geometric.utils import loop
import torch.optim as optim
from utils import load_data, load_data_old, accuracy, condition_number
from models.gcn_conv import GCN_node_classification
from models.conv_only import Conv_Only
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import networkx as nx
from numpy import dot
import wandb
import sys
import time


import torch

def compute_knn_graph(features: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Constructs a k-NN graph adjacency matrix efficiently.
    Uses torch.cdist to avoid OOM errors on large graphs.
    """
    # 1. Compute pairwise distance efficiently
    # torch.cdist computes the L2 distance (Euclidean)
    # Memory complexity: O(N^2) instead of O(N^2 * d)
    dist = torch.cdist(features, features, p=2)
    
    # We square it to match the standard ||x - y||^2 metric usually expected
    dist_sq = dist.pow(2)
    
    # 2. Get k-Nearest Neighbors
    # We select k+1 because the closest neighbor is the node itself (dist=0)
    # largest=False finds the smallest distances
    _, indices = dist_sq.topk(k + 1, largest=False, sorted=True)

    # 3. Construct Adjacency Matrix W
    N = features.size(0)
    
    # Remove the first column (self-loops)
    neighbor_indices = indices[:, 1:]
    
    # Create row indices efficiently
    row_indices = torch.arange(N, device=features.device).unsqueeze(1).expand(-1, k)
    
    # Initialize sparse-like adjacency (dense tensor for now)
    W = torch.zeros((N, N), device=features.device)
    W[row_indices, neighbor_indices] = 1.0
    
    # Symmetrize (Critical for valid spectral analysis)
    W = torch.max(W, W.t())
    
    return W


def compute_normalized_laplacian(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes L_sym = I - D^-1/2 W D^-1/2. 
    Normalized Laplacians are more stable for spectral comparison[cite: 290].
    """
    N = W.size(0)
    d = W.sum(dim=1)
    d_inv_sqrt = torch.pow(d + eps, -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    
    L_norm = torch.eye(N, device=W.device) - D_inv_sqrt @ W @ D_inv_sqrt
    return L_norm

def spade_score(X: torch.Tensor, SX: torch.Tensor, k: int = 5, reg: float = 1e-3) -> float:
    """
    Computes Adapted SPADE Score using Normalized Laplacians and Mean Distortion.
    """
    # 1. Construct Graphs
    W_X = compute_knn_graph(X, k=k)
    W_SX = compute_knn_graph(SX, k=k)
    
    # 2. Compute Normalized Laplacians for stability [cite: 290]
    L_X = compute_normalized_laplacian(W_X)
    L_SX = compute_normalized_laplacian(W_SX)
    
    # 3. Robust Generalized Eigenvalue Handling
    N = X.size(0)
    # Use a larger, more robust regularization for the denominator matrix 
    L_X_reg = L_X + reg * torch.eye(N, device=X.device)
    
    try:
        # Whitening transformation: S = L_inv @ L_SX @ L_inv.T
        L_chol = torch.linalg.cholesky(L_X_reg)
        L_inv = torch.inverse(L_chol)
        S = L_inv @ L_SX @ L_inv.t()
    except RuntimeError:
        # Fallback to Pseudo-inverse if matrix is still singular 
        L_X_inv = torch.linalg.pinv(L_X_reg)
        # Approximate whitening using sqrt of pinv
        S = L_X_inv @ L_SX 

    # 4. Compute Eigenvalues
    # We use eigvals (non-symmetric solver) as fallback if S isn't perfectly symmetric
    try:
        eigenvalues = torch.linalg.eigvalsh(S)
    except RuntimeError:
        eigenvalues = torch.linalg.eigvals(S).real

    # ADAPTATION: Return the mean of top eigenvalues (Average Distortion)
    # This correlates better with accuracy than the maximum.
    return eigenvalues.mean().item()


########################################################################################
# Parse arguments 
########################################################################################

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='./data/',  help='Directory of datasets; default is ./data/')
parser.add_argument('--k', type=int, default=5,  help='The number of neighboors k in the KNN construction of the graph')
parser.add_argument('--gso', type=str, default='NORMADJ',  choices=['ADJ', 'UNORMLAPLACIAN','SINGLESSLAPLACIAN','RWLAPLACIAN','SYMLAPLACIAN','NORMADJ', 'MEANAGG', 'Test'])
parser.add_argument('--outdir', type=str, default='./exps/', help='Directory of experiments output; default is ./exps/')
parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name; default is Cora' ,choices=["Cora" ,"CiteSeer", "PubMed", "CS", "ogbn-arxiv", 'Reddit', 'genius', 'Penn94','Computers','Photo', "Physics", 'deezer-europe', 'arxiv-year' , "imdb" , "chameleon", "crocodile", "squirrel","Cornell", "Texas", "Wisconsin"])
parser.add_argument('--device', type=int, default=1,help='Set CUDA device number; if set to -1, disables cuda.')    
parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
parser.add_argument('--use_wandb', type= bool,default = False , choices=[True, False])
args = parser.parse_args()

device = torch.device('cuda:'+str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

expfolder = osp.join(args.outdir, args.dataset)
expfolder = osp.join(expfolder,datetime.now().strftime("%Y%m%d_%H%M%S"))
Path(expfolder).mkdir(parents=True, exist_ok=True)

########################################################################################
# Data loading and model setup 
########################################################################################
all_test_acc = []
adj, features, labels, idx_train, idx_val, idx_test = load_data(path = args.datadir, dataset_name = args.dataset,device =  device)

# !!!!!!!!
# !!!!!!!!
# ADDING self loops is learnable in the PGSO
n = features.size(0)
#  adj = adj + sp.identity(n)

edge_index_without_loops, edge_weight_without_loops = from_scipy_sparse_matrix(adj)
edge_index_without_loops = edge_index_without_loops.to(features.device)

row, col = edge_index_without_loops
deg = degree(col, n, dtype=features.dtype)
diags = deg



    # Now that we have computed the centralities without self loops, we add self loops in the edge index

edge_index, edge_weight = from_scipy_sparse_matrix(adj)
edge_index = edge_index.to(features.device)
row, col = edge_index
diags = diags.clone()
edge_index_id, edge_weight_id = from_scipy_sparse_matrix(sp.identity(n))
edge_index_id = edge_index_id .to(features.device)
norm_diag  = torch.tensor([diags[edge_index[0,i].item()].item() if edge_index[0,i].item() == edge_index[1,i].item() else 0 for i in range(edge_index.size(-1))  ]).to(diags.device)
    # norm_identity  = torch.tensor([1 if edge_index[0,i].item() == edge_index[1,i].item() else 0 for i in range(edge_index.size(-1))  ]).to(diags.device)

# Model and optimizer
model = Conv_Only( gso = args.gso).to(device)

# Freezing the weights of the PGSO
model.m1.requires_grad = False
model.m2.requires_grad = False
model.m3.requires_grad = False
model.e1.requires_grad = False
model.e2.requires_grad = False
model.e3.requires_grad = False
model.a.requires_grad = False

# Apply Conv
gso_output = model(features.float(), edge_index.detach() ,edge_index_id.detach(), diags.float()) 
num_classes = labels.max().item() + 1
one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()

# Compute Spade
ts = time.time()
score = spade_score(one_hot_labels, gso_output, k=args.k)
tf = time.time()
print(f"SPADE Score (Alignment Distortion): {score:.4f}")

