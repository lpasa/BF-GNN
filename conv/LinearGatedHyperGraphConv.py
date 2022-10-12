from torch_geometric.typing import OptPairTensor
from torch import Tensor
from torch import  randint
from conv.LinearGatedHyper import LinearGatedHyper
from torch_sparse import matmul
from torch_geometric.nn.conv import MessagePassing

class LinearGatedHyperGraphConv(MessagePassing):

    def __init__(self, in_channels, out_channels, context_dim, n_hyperplanes, aggr = 'add', **kwargs):
        super(LinearGatedHyperGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_hyperplanes = n_hyperplanes

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = LinearGatedHyper(in_channels[0], out_channels, context_dim, n_hyperplanes)
        self.lin_r = LinearGatedHyper(in_channels[1], out_channels, context_dim, n_hyperplanes)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def set_prototypes(self, training_set,protopytes_indices=None, ):
        if protopytes_indices is None or protopytes_indices.shape[0] != self.out_channels:
            protopytes_indices = randint(low=0, high=training_set.shape[0], size=(self.out_channels,self.n_prototypes))
        self.lin_l.set_prototypes(training_set[protopytes_indices])
        self.lin_r.set_prototypes(training_set[protopytes_indices])


    def forward(self, x, context, edge_index,
                edge_weight = None, size = None):

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_l(out, context)
        x_r = x[1]
        if x_r is not None :
            out += self.lin_r(x_r, context)
        return out


    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
