from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch import  randint
from torch_sparse import SparseTensor
from torch.nn import Parameter
import torch
from conv.LinearGatedProto import LinearGatedProto

class LinearGatedProtoGCNConv(MessagePassing):


    def __init__(self, in_channels,
                 out_channels,
                 context_dim,
                 n_prototypes,
                 improved= False,
                 cached = False,
                 add_self_loops = True,
                 normalize = True):

        super(LinearGatedProtoGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.n_prototypes = n_prototypes

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.linear_proj = LinearGatedProto(in_channels, out_channels, context_dim, n_prototypes)


        self.reset_parameters()

    def reset_parameters(self):
        self.linear_proj.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None


    def set_prototypes(self, training_set,protopytes_indices=None):
        if protopytes_indices is None or protopytes_indices.shape[0] != self.out_channels:
            protopytes_indices = randint(low=0, high=training_set.shape[0], size=(self.out_channels,self.n_prototypes))
        self.linear_proj.set_prototypes(training_set[protopytes_indices])


    def forward(self, x,context, edge_index,
                edge_weight = None) :

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.linear_proj(x, context)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j, edge_weight) :
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x) :
        return torch.matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)