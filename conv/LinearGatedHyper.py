from torch.nn.parameter import Parameter
from torch.nn.functional import normalize
from torch.nn.functional import one_hot
from torch.nn import  init
import math
import torch


class LinearGatedHyper(torch.nn.Module):
    def __init__(self, in_features, out_features, context_dim, n_hyperplanes, hyp_bias_std=0.05,device=None):
        super(LinearGatedHyper, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.context_dim = context_dim
        self.n_hyperplanes = n_hyperplanes
        self.weights = Parameter(torch.zeros(self.out_features, 2 ** self.n_hyperplanes, self.in_features).to(self.device))
        self.bias = Parameter(torch.zeros(self.out_features, 2 ** self.n_hyperplanes).to(self.device))

        self.index_tensor = torch.zeros(self.out_features, dtype=torch.int64).to(self.device)
        for i in range(self.out_features):
            self.index_tensor[i] += i * 2 ** n_hyperplanes

        self.hyp_w = torch.normal(0.0, 1.0, size=(self.out_features, n_hyperplanes, self.context_dim)).to(self.device)
        self.hyp_w = normalize(self.hyp_w, dim=2)
        self.hyp_b = torch.normal(0.0, hyp_bias_std, size=(self.out_features, n_hyperplanes, 1)).to(self.device)
        self.hyp_w.requires_grad=False
        self.hyp_b.requires_grad=False

        self.bit_importance = torch.zeros(n_hyperplanes).to(self.device)
        for i in range(n_hyperplanes):
            self.bit_importance[i] = 2**(i)  # [2**n_hyp, ... , 1]

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, context, min_val=-2, max_val=2):
        """
        Works only with batch size of 1 (online)
        :param x: input of shape [batch_size, in_features, n_classes]
        :param side_info: side information of shape [batch_size, side_info_dim]
        :return:
        """
        # input clipping
        x = torch.sigmoid(x)
        x = torch.logit(x, 0.01)

        #compute the out for all the weights
        all_out = torch.matmul(self.weights, x.T).permute(2, 0, 1) + self.bias.repeat(x.shape[0], 1, 1)

        #find the index of the selected weights base on the hyperplane
        cx_w_b=torch.gt(torch.matmul(self.hyp_w, context.T), self.hyp_b)
        index_map = torch.matmul(cx_w_b.permute(0, 2, 1).to(torch.float32), self.bit_importance)
        #create mask and apply
        gate_tensor = one_hot(index_map.T.to(torch.int64), 2 ** self.n_hyperplanes).to(self.device)
        gated_context_output = all_out * gate_tensor
        output = torch.sum(gated_context_output,dim=2)
        output = torch.clip(output, min_val, max_val)

        return output

