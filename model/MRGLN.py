import torch
torch.set_printoptions(profile="full")
from torch_geometric.utils.get_laplacian import get_laplacian
from utils.utils_method import get_SP_from_batch, get_k_SP
from layer.KGatedLayer import KGateLayer
import os
import datetime


class MRGLN(torch.nn.Module):
    def __init__(self, in_channels, n_class=2, k=3, diffusion_op='A', n_proto=2, GLN_type="proto", device=None):
        '''

        :param in_channels:
        :param n_class:
        :param drop_prob:
        :param k:
        :param output:
        :param diffusion_op: {A,L,S} respectively Adjacency Matrix, Laplacian Matrix, Sortetest Past
        :param device:
        '''
        super(MRGLN, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels = in_channels
        self.n_class = n_class
        self.k = k
        self.diffusion_op = diffusion_op
        self.diffusion_op_getter = self.get_diffusion_op_fun()
        self.diffusion_op_step = self.get_diffusion_op_step_fun()
        self.n_prototypes=n_proto

        self.hidden_gated_neurons= torch.nn.ModuleList()

        for i in range(k):

            self.hidden_gated_neurons.append(KGateLayer(in_channels=in_channels*(i+1),
                                                        context_dim=in_channels,
                                                        n_class=n_class,
                                                        n_proto=n_proto,
                                                        GLN_type=GLN_type,
                                                        device=self.device))

        self.out_fun = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        pass


    @staticmethod
    def get_A(data,device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
           device = device
        adjacency_indexes = data.edge_index
        A_rows = adjacency_indexes[0]
        A_data = [1] * A_rows.shape[0]
        v_index = torch.FloatTensor(A_data).to(device)
        A_shape = [data.x.shape[0], data.x.shape[0]]
        A = torch.sparse.FloatTensor(adjacency_indexes, v_index, torch.Size(A_shape)).to_dense()
        return A

    @staticmethod
    def get_L(data):
        L_edge_index, L_values = get_laplacian(data.edge_index, normalization="sym")
        L = torch.sparse.FloatTensor(L_edge_index, L_values, torch.Size([data.x.shape[0], data.x.shape[0]])).to_dense()
        return L

    @staticmethod
    def get_S(data):
        return get_SP_from_batch(data).float()


    def get_diffusion_op_fun(self):
        if self.diffusion_op == 'A':
            return self.get_A
        elif self.diffusion_op == 'L':
            return self.get_L
        elif self.diffusion_op == 'S':
            return self.get_S


    @staticmethod
    def get_power_series_step(M,i):
        return torch.matrix_power(M, i)

    @staticmethod
    def get_S_step(S,i):
        return get_k_SP(S, i)


    def get_diffusion_op_step_fun(self):
        if self.diffusion_op == 'A':
            return self.get_power_series_step
        elif self.diffusion_op == 'L':
            return self.get_power_series_step
        elif self.diffusion_op == 'S':
            return self.get_S_step

    def set_networks_layer_protopytes_based_on_x(self, data, mask):
        for k_layer in self.hidden_gated_neurons:
            k_layer.set_networks_layer_protopytes_based_on_x(data,mask)

    def set_networks_optimizer(self,lr,weight_decay,criterion):
        for k_layer in self.hidden_gated_neurons:
            k_layer.set_layer_optimizer(lr,weight_decay,criterion)

    def PLGFNN_optimization_and_test(self, data, train_mask, test_mask, valid_mask, context, epochs, test_name="",
                                 log_path=".", max_n_epochs_without_improvements=30, early_stopping_threshold=0,
                                 test_epoch=1):

        X = data.x.to(self.device)
        y = data.y
        diffusion_op = self.diffusion_op_getter(data)

        H = [X]
        for i in range(1, self.k):
            xhi_layer_i = torch.mm(self.diffusion_op_step(diffusion_op, i), X)
            H.append(xhi_layer_i)

        H = torch.cat(H, dim=1)


        Gated_H=[]
        for k, k_layer in enumerate(self.hidden_gated_neurons):
            current_input=H[:,0:self.in_channels*(k+1)]

            print("Optimization layer of k = ",k)

            k_layer.gated_layer_optimization_step(X=current_input,
                                                  y=y,
                                                  train_mask=train_mask,
                                                  test_mask=test_mask,
                                                  valid_mask=valid_mask,
                                                  context=context,
                                                  epochs=epochs,
                                                  test_name=test_name+"_k_layer_"+str(k),
                                                  log_path=log_path,
                                                  max_n_epochs_without_improvements=max_n_epochs_without_improvements,
                                                  early_stopping_threshold=early_stopping_threshold,
                                                  test_epoch=test_epoch
                                                  )
            Gated_H.append(torch.cat(k_layer(current_input,context),dim=1))

        Gated_H = torch.stack(Gated_H)
        Gated_H = Gated_H.permute(1, 0, 2)
        #Geometric mean
        Gated_H = self.out_fun(Gated_H)
        prob_classes = torch.pow(torch.prod(Gated_H, 1), 1 / (k + 1))

        # eval model
        acc_train, _, _ = self.eval_model(prob_classes, data.y, train_mask)
        acc_test, _, _ = self.eval_model(prob_classes, data.y, test_mask)
        acc_valid, _, _ = self.eval_model(prob_classes, data.y, valid_mask)

        results_file = open(os.path.join(log_path, (test_name + "_results")), 'w+')
        results_file.write("test_name: %s \n" % test_name)
        results_file.write(str(datetime.datetime.now()) + '\n')
        results_file.write("acc training \t  acc test \t acc valid \n")
        results_file.write("{: .8f}\t{: .8f}\t{: .8f}\n".format(acc_train, acc_test, acc_valid))

        print(" acc training:{: .8f}\n acc test: {: .8f}\n acc valid: {: .8f}\n".format(acc_train, acc_test, acc_valid))
        print("------------------------")
        # input("DEBUG")


    def eval_model(self, prob, y, mask):
        pred = prob.max(1)[1]
        n_samples = len(y[mask])
        correct = pred[mask].eq(y[mask]).sum().item()
        acc = correct / n_samples
        return acc, correct, n_samples

    def save_model(self,test_name, log_folder='./'):
        torch.save(self.state_dict(), os.path.join(log_folder,test_name+'.pt'))





