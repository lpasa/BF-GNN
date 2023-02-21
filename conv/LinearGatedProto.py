from torch.nn.parameter import Parameter
from torch.nn.functional import one_hot
from torch.nn.functional import normalize
from torch.nn import  init
import math
import torch



class LinearGatedProto(torch.nn.Module):
    def __init__(self,in_features, out_features, context_dim, n_prototypes):
        super(LinearGatedProto, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.context_dim = context_dim
        self.n_prototypes = n_prototypes
        self.weight = Parameter(torch.FloatTensor(out_features, n_prototypes, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features, n_prototypes))
        self.prototypes = torch.rand(out_features,n_prototypes, context_dim) #each neuron of the layer have it own prototypes


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def set_prototypes(self,prototypes):
        self.prototypes=prototypes


    def forward(self, X, context, min_val=-100, max_val=100):

        all_context_output = torch.matmul(self.weight, X.float().T)
        all_context_output = all_context_output.permute(2,0,1)+self.bias.repeat(X.shape[0],1,1)

        all_dist = torch.cdist(self.prototypes.to(context.device),context,p=2).permute(2,0,1)
        best_proto=torch.argmin(all_dist,dim=-1)
        gate_tensor = one_hot(best_proto, self.n_prototypes)
        gated_context_output = all_context_output * gate_tensor
        gated_context_output = torch.sum(gated_context_output,dim=2)

        return gated_context_output