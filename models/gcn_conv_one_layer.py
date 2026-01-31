################################
# Convolutional models
################################
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GCNConv
from torch_geometric.nn import global_add_pool, GINConv
from torch.nn.init import xavier_uniform_
import torch
import numpy as np
from utils import condition_number
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import dense_to_sparse
import networkx as nx
import sys
from typing import Optional

def bjorck_orthogonalize(w: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """
    Ensures the weight matrix W is orthogonal (W^T W = I).
    This prevents the model from stretching or compressing subspaces to
    'fix' a poor GSO, making the accuracy/MSD comparison fair.
    """
    # Scaling to ensure the singular values are <= 1 for convergence
    # Using a safety margin of 1.1 for the spectral norm
    w_norm = torch.linalg.matrix_norm(w, ord=2)
    w = w / (w_norm * 1.1)

    for _ in range(iterations):
        w_t_w = torch.mm(w, w.t()) if w.size(0) < w.size(1) else torch.mm(w.t(), w)
        # Standard Björck update: W = W * (1.5 * I - 0.5 * W^T * W)
        # This implementation uses the iterative series expansion
        if w.size(0) < w.size(1):
            w = 1.5 * w - 0.5 * torch.mm(w_t_w, w)
        else:
            w = 1.5 * w - 0.5 * torch.mm(w, w_t_w)
    return w



class GCN_BJORK(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, gso):
        super(GCN_BJORK, self).__init__()
        self.conv_layers = nn.ModuleList(
            [GCNConv(input_dim, output_dim)] 
        )

        self.dropout = dropout
        if gso == 'ADJ' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso == 'UNORMLAPLACIAN' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'e1'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso == 'SINGLESSLAPLACIAN' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e1'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso == 'RWLAPLACIAN' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'm3' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2'  : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso == 'SYMLAPLACIAN' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'm3' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'e3'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

        if gso == 'NORMADJ' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'e3'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'a'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

        if gso == 'MEANAGG' :
            for param in ['m1', 'm2', "m3", 'e1', 'e2','e3','a']:
                if param == 'm2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

    def compute_gso_1(self,row, col, diags):

       # Compute the term m2* (diag**e2)A(diag**e3)
        diags_pow_e2 = diags.pow(self.e2)
        diags_pow_e3 = diags.pow(self.e3)
        diags_pow_e2[diags_pow_e2 ==float('inf')] = 0
        diags_pow_e3[diags_pow_e3 ==float('inf')] = 0
        norm_normalization = diags_pow_e2[row] * diags_pow_e3[col]
        # norm_normalization = norm_normalization * (1-norm_identity)
        gso_1 = self.m2 * norm_normalization 

        return gso_1
    
    def compute_gso_2(self,row_id, col_id, diags):
        # Compute the term m1* (diag**e1)
        #norm_1 = norm_diag.pow(self.e1)
        diags_pow_e1 = diags.pow(self.e1)
        diags_pow_e1[diags_pow_e1 ==float('inf')] = 0
        norm_1 = diags_pow_e1[row_id]
        norm_1 = self.m1 * norm_1

        # Compute the term:  m2*a* (diag**e2)I(diag**e3)
        diags_pow_e2 = diags.pow(self.e2)
        diags_pow_e3 = diags.pow(self.e3)
        diags_pow_e2[diags_pow_e2 ==float('inf')] = 0
        diags_pow_e3[diags_pow_e3 ==float('inf')] = 0
        norm_normalization = diags_pow_e2[row_id] * diags_pow_e3[col_id]
        norm_2_l = self.m2 * norm_normalization 

        
        # Compute the first term m3*I
        norm_3 = self.m3

        # Final Norm
        gso_2 = norm_1 + norm_2_l + norm_3
        return gso_2
    
    def forward(self, x, edge_index , edge_index_id, diags):
        row, col = edge_index
        row_id, col_id = edge_index_id
        gso_1 = self.compute_gso_1(row, col, diags)
        gso_2 = self.compute_gso_2(row_id, col_id, diags)
        h = x
        for i, layer in enumerate(self.conv_layers):
            original_weight = layer.lin.weight
            layer.lin.weight.data = bjorck_orthogonalize(original_weight)
            h1 = layer(h, edge_index , gso_1)
            h2 = layer(h, edge_index_id , gso_2)
            h = h1 + h2
            if i < len(self.conv_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)
        return F.log_softmax(h, dim=1)


