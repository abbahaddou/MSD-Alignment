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





class GCN_Two_GSO(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, gso_1, gso_2):
        super(GCN_Two_GSO, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList(
            [GCNConv(input_dim, hidden_dim)] +
            [GCNConv(hidden_dim, hidden_dim) for _ in range(1, num_layers-1)] + 
            [GCNConv(hidden_dim, output_dim)]
        )

        self.dropout = dropout
        if gso_1 == 'ADJ' :
            for param in ['m1_1', 'm2_1', "m3_1", 'e1_1', 'e2_1','e3_1','a_1']:
                if param == 'm2_1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso_1 == 'UNORMLAPLACIAN' :
            for param in ['m1_1', 'm2_1', "m3_1", 'e1_1', 'e2_1','e3_1','a_1']:
                if param == 'm1_1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'm2_1' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'e1_1'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso_1 == 'SINGLESSLAPLACIAN' :
            for param in ['m1_1', 'm2_1', "m3_1", 'e1_1', 'e2_1','e3_1','a_1']:
                if param == 'm1_1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'm2_1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e1_1'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso_1 == 'RWLAPLACIAN' :
            for param in ['m1_1', 'm2_1', "m3_1", 'e1_1', 'e2_1','e3_1','a_1']:
                if param == 'm2_1' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'm3_1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2_1'  : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso_1 == 'SYMLAPLACIAN' :
            for param in ['m1_1', 'm2_1', "m3_1", 'e1_1', 'e2_1','e3_1','a_1']:
                if param == 'm2_1' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'm3_1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2_1'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'e3_1'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

        if gso_1 == 'NORMADJ' :
            for param in ['m1_1', 'm2_1', "m3_1", 'e1_1', 'e2_1','e3_1','a_1']:
                if param == 'm2_1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2_1'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'e3_1'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'a_1'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

        if gso_1 == 'MEANAGG' :
            for param in ['m1_1', 'm2_1', "m3_1", 'e1_1', 'e2_1','e3_1','a_1']:
                if param == 'm2_1' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2_1' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))









        if gso_2 == 'ADJ' :
            for param in ['m1_2', 'm2_2', "m3_2", 'e1_2', 'e2_2','e3_2','a_2']:
                if param == 'm2_2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso_2 == 'UNORMLAPLACIAN' :
            for param in ['m1_2', 'm2_2', "m3_2", 'e1_2', 'e2_2','e3_2','a_2']:
                if param == 'm1_2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'm2_2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'e1_2'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso_2 == 'SINGLESSLAPLACIAN' :
            for param in ['m1_2', 'm2_2', "m3_2", 'e1_2', 'e2_2','e3_2','a_2']:
                if param == 'm1_2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'm2_2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e1_2'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso_2 == 'RWLAPLACIAN' :
            for param in ['m1_2', 'm2_2', "m3_2", 'e1_2', 'e2_2','e3_2','a_2']:
                if param == 'm2_2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'm3_2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2_2'  : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
        if gso_2 == 'SYMLAPLACIAN' :
            for param in ['m1_2', 'm2_2', "m3_2", 'e1_2', 'e2_2','e3_2','a_2']:
                if param == 'm2_2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                elif param == 'm3_2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2_2'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'e3_2'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

        if gso_2 == 'NORMADJ' :
            for param in ['m1_2', 'm2_2', "m3_2", 'e1_2', 'e2_2','e3_2','a_2']:
                if param == 'm2_2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2_2'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'e3_2'  : 
                    self.register_parameter(param, nn.Parameter(-(1/2)*torch.ones([1])))
                elif param == 'a_2'  : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

        if gso_2 == 'MEANAGG' :
            for param in ['m1_2', 'm2_2', "m3_2", 'e1_2', 'e2_2','e3_2','a_2']:
                if param == 'm2_2' : 
                    self.register_parameter(param, nn.Parameter(torch.ones([1])))
                elif param == 'e2_2' : 
                    self.register_parameter(param, nn.Parameter(-torch.ones([1])))
                else : 
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

    def compute_gso1_1(self,row, col, diags):

       # Compute the term m2* (diag**e2)A(diag**e3)
        diags_pow_e2 = diags.pow(self.e2_1)
        diags_pow_e3 = diags.pow(self.e3_1)
        diags_pow_e2[diags_pow_e2 ==float('inf')] = 0
        diags_pow_e3[diags_pow_e3 ==float('inf')] = 0
        norm_normalization = diags_pow_e2[row] * diags_pow_e3[col]
        # norm_normalization = norm_normalization * (1-norm_identity)
        gso_1 = self.m2_1 * norm_normalization 

        return gso_1
    
    def compute_gso1_2(self,row_id, col_id, diags):
        # Compute the term m1* (diag**e1)
        #norm_1 = norm_diag.pow(self.e1)
        diags_pow_e1 = diags.pow(self.e1_1)
        diags_pow_e1[diags_pow_e1 ==float('inf')] = 0
        norm_1 = diags_pow_e1[row_id]
        norm_1 = self.m1_1 * norm_1

        # Compute the term:  m2*a* (diag**e2)I(diag**e3)
        diags_pow_e2 = diags.pow(self.e2_1)
        diags_pow_e3 = diags.pow(self.e3_1)
        diags_pow_e2[diags_pow_e2 ==float('inf')] = 0
        diags_pow_e3[diags_pow_e3 ==float('inf')] = 0
        norm_normalization = diags_pow_e2[row_id] * diags_pow_e3[col_id]
        norm_2_l = self.m2_1 * norm_normalization 

        
        # Compute the first term m3*I
        norm_3 = self.m3_1

        # Final Norm
        gso_2 = norm_1 + norm_2_l + norm_3
        return gso_2




    def compute_gso2_1(self,row, col, diags):

       # Compute the term m2* (diag**e2)A(diag**e3)
        diags_pow_e2 = diags.pow(self.e2_2)
        diags_pow_e3 = diags.pow(self.e3_2)
        diags_pow_e2[diags_pow_e2 ==float('inf')] = 0
        diags_pow_e3[diags_pow_e3 ==float('inf')] = 0
        norm_normalization = diags_pow_e2[row] * diags_pow_e3[col]
        # norm_normalization = norm_normalization * (1-norm_identity)
        gso_1 = self.m2_2 * norm_normalization 

        return gso_1
    
    def compute_gso2_2(self,row_id, col_id, diags):
        # Compute the term m1* (diag**e1)
        #norm_1 = norm_diag.pow(self.e1)
        diags_pow_e1 = diags.pow(self.e1_2)
        diags_pow_e1[diags_pow_e1 ==float('inf')] = 0
        norm_1 = diags_pow_e1[row_id]
        norm_1 = self.m1_2 * norm_1

        # Compute the term:  m2*a* (diag**e2)I(diag**e3)
        diags_pow_e2 = diags.pow(self.e2_2)
        diags_pow_e3 = diags.pow(self.e3_2)
        diags_pow_e2[diags_pow_e2 ==float('inf')] = 0
        diags_pow_e3[diags_pow_e3 ==float('inf')] = 0
        norm_normalization = diags_pow_e2[row_id] * diags_pow_e3[col_id]
        norm_2_l = self.m2_2 * norm_normalization 

        
        # Compute the first term m3*I
        norm_3 = self.m3_2

        # Final Norm
        gso_2 = norm_1 + norm_2_l + norm_3
        return gso_2
    
    def forward(self, x, edge_index , edge_index_id, diags):
        row, col = edge_index
        row_id, col_id = edge_index_id
        gso1_1 = self.compute_gso1_1(row, col, diags)
        gso1_2 = self.compute_gso1_2(row_id, col_id, diags)

        gso2_1 = self.compute_gso2_1(row, col, diags)
        gso2_2 = self.compute_gso2_2(row_id, col_id, diags)
        h = x
        for i, layer in enumerate(self.conv_layers):
            if i == 0: 
                h1 = layer(h, edge_index , gso1_1)
                h2 = layer(h, edge_index_id , gso1_2)
            else:
                h1 = layer(h, edge_index , gso2_1)
                h2 = layer(h, edge_index_id , gso2_2)
            h = h1 + h2
            if i < len(self.conv_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)
        return F.log_softmax(h, dim=1)


