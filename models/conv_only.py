import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    pass


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    pass


def gcn_norm(  # noqa: F811
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes = None,
    improved = False,
    add_self_loops = True,
    flow: str = "source_to_target",
    dtype = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class Conv_Scratch(MessagePassing):

    def __init__(
        self,
        improved = False,
        cached = False,
        add_self_loops = None,
        normalize = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")


        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None




    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache


        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class Conv_Only(nn.Module):
    def __init__(self,  gso):
        super(Conv_Only, self).__init__()
        self.layer = Conv_Scratch()

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
        h1 = self.layer(h, edge_index , gso_1)
        h2 = self.layer(h, edge_index_id , gso_2)
        h = h1 + h2
        return h
