from conv.LinearGatedProto import LinearGatedProto
from conv.LinearGatedHyper import LinearGatedHyper
from conv.LinearGatedProtoGraphConv import LinearGatedGraphConvProto
import torch
torch.set_printoptions(profile="full")


class MR_ClassLayer(torch.nn.Module):

    def __init__(self, in_channels, n_neuron,context_dim, n_prototypes, layer_class,conv_type=None, device=None):
        super(MR_ClassLayer, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels = in_channels
        self.n_neuron = n_neuron
        self.n_prototypes=n_prototypes
        self.contex_dim = context_dim
        self.layer_class= layer_class
        self.conv_type = conv_type
        if conv_type == "proto" or None:
            self.layer = LinearGatedProto(in_features=in_channels,
                                          out_features=n_neuron,
                                          context_dim=context_dim,
                                          n_prototypes=n_prototypes)
        elif conv_type == "hyper":
            self.layer = LinearGatedHyper(in_features= in_channels,
                                          out_features=n_neuron,
                                          context_dim=context_dim,
                                          n_hyperplanes=n_prototypes)

        self.out_fun = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

    def set_prototypes(self, training_set,protopytes_indices):
        if self.conv_type == "proto" or None:
            self.layer.set_prototypes(training_set[protopytes_indices])



    def forward(self, X, context):
        h=self.layer(X,context)
        return h

    def set_opt(self,lr,weight_decay,criterion):
         self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
         self.criterion=criterion

    def opt_step(self, h, mask, context, y):
            self.optimizer.zero_grad()

            h = self.forward(h, context)
            pred_c = self.out_fun(h)

            #------#
            target = torch.where(y==self.layer_class,1,0).reshape(-1,1).expand(-1,self.n_neuron)
            target=target.float()
            #-------#

            loss = self.criterion((pred_c[mask]), target[mask])
            loss.backward()
            self.optimizer.step()

            return loss, h.detach()


    #Layer Evaluation
    def eval_layer(self, h,y ,mask, context):
        self.eval()
        h = self.forward(h, context)
        pred = self.out_fun(h)[mask]
        # ------#
        target = torch.where(y == self.layer_class, 1, 0).reshape(-1, 1).expand(-1, self.n_neuron)
        target = target[mask]
        n_samples = len(target)* self.n_neuron
        round_pred=torch.round(pred)
        correct = round_pred.eq(target).sum().item()
        acc = correct / n_samples
        loss = self.criterion(pred, target.float()).item()

        return acc, correct, n_samples, loss / n_samples

class MP_ClassLayer(MR_ClassLayer):
    def __init__(self, in_channels, n_neuron,context_dim, n_prototypes, layer_class, conv=None,conv_act=lambda x: x,
                 device=None):
        super(MR_ClassLayer, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels = in_channels
        self.n_neuron = n_neuron
        self.n_prototypes=n_prototypes
        self.contex_dim = context_dim
        self.conv_act = conv_act
        self.layer_class= layer_class

        if conv is None:

            self.layer = LinearGatedGraphConvProto(in_channels=in_channels,
                                                   out_channels=n_neuron,
                                                   context_dim=context_dim,
                                                   n_prototypes=n_prototypes)
        else:
            if 'Hyper' in conv.__name__:
                self.layer = conv(in_channels=in_channels,
                                  out_channels=n_neuron,
                                  context_dim=context_dim,
                                  n_hyperplanes=n_prototypes)
            else:
                self.layer=conv(in_channels=in_channels,
                                out_channels=n_neuron,
                                context_dim=context_dim,
                                n_prototypes=n_prototypes)
        self.out_fun = torch.nn.Sigmoid()
        self.reset_parameters()

    def set_prototypes(self, training_set,protopytes_indices=None):
        self.layer.set_prototypes(training_set,protopytes_indices)

    def opt_step(self, h, mask, context, y, edge_index):
            self.optimizer.zero_grad()
            h = self.forward(h, context, edge_index)
            pred_c = self.out_fun(h)
            target = torch.where(y==self.layer_class,1,0).reshape(-1,1).expand(-1,self.n_neuron)
            target=target.float()
            loss = self.criterion((pred_c[mask]), target[mask])
            loss.backward()
            self.optimizer.step()
            return loss, h.detach()

    def forward(self, X, context, edge_index):
        h = self.layer(X, context, edge_index)
        return h

    def eval_layer(self, h, edge_index, y, mask, context):
        self.eval()
        h = self.forward(h, context, edge_index)
        pred = self.out_fun(h)[mask]
        # ------#
        target = torch.where(y == self.layer_class, 1, 0).reshape(-1, 1).expand(-1, self.n_neuron)
        target = target[mask]
        n_samples = len(target) * self.n_neuron
        round_pred = torch.round(pred)
        correct = round_pred.eq(target).sum().item()
        acc = correct / n_samples
        loss = self.criterion(pred, target.float()).item()

        return acc, correct, n_samples, loss / n_samples
