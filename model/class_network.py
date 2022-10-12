import os
import datetime
import torch
from layer.layer_class import MP_ClassLayer
import time
import numpy as np

def prepare_log_files(test_name, log_dir):
    train_log = open(os.path.join(log_dir, (test_name + "_train")), 'w+')
    test_log = open(os.path.join(log_dir, (test_name + "_test")), 'w+')
    valid_log = open(os.path.join(log_dir, (test_name + "_valid")), 'w+')

    for f in (train_log, test_log, valid_log):
        f.write("test_name: %s \n" % test_name)
        f.write(str(datetime.datetime.now()) + '\n')
        f.write("#epoch \t layer \t loss \t acc \t avg_epoch_time \n")

    return train_log, test_log, valid_log

class MPClassNetwork(torch.nn.Module):
    def __init__(self, in_channels, n_neuron,context_dim, n_prototypes,network_class, n_layer, conv=None, device=None ,input_as_context = False):
        super(MPClassNetwork, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels = in_channels
        self.n_neuron = n_neuron
        self.n_prototypes = n_prototypes
        self.contex_dim = context_dim
        self.n_layer = n_layer
        self.network_class=network_class
        self.layers = torch.nn.ModuleList()

        #Define layers
        self.layers.append(
            MP_ClassLayer(in_channels, n_neuron[0], context_dim, n_prototypes, network_class, conv, device=device))
        self.protopytes_indices = [None]
        for i in range(1, n_layer-1):

            if input_as_context is False:
                self.layers.append(
                    MP_ClassLayer(n_neuron[i - 1], n_neuron[i], n_neuron[i - 1], n_prototypes, network_class, conv, device=device))
            else:
                self.layers.append(
                    MP_ClassLayer(n_neuron[i - 1], n_neuron[i], self.in_channels, n_prototypes, network_class, conv,
                                  device=device))
            self.protopytes_indices.append(None)

        #last layer
        if input_as_context is False:
            self.layers.append(MP_ClassLayer(n_neuron[-1], 1, n_neuron[-1], n_prototypes, network_class, conv, device=device))
        else:
            self.layers.append(
                MP_ClassLayer(n_neuron[-1], 1, self.in_channels, n_prototypes, network_class, conv, device=device))
        self.protopytes_indices.append(None)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def set_layers_prototypes_base_on_input_rep(self, data, mask,fix_protopytes_indices=False):
        for l,layer in enumerate(self.layers):
            if fix_protopytes_indices:
                if self.protopytes_indices[l] is None:
                    self.protopytes_indices[l]= torch.randint(low=0, high=data.x[mask].shape[0], size=(self.n_neuron[l],self.n_prototypes))
            layer.set_prototypes(data.x, self.protopytes_indices[l])

    def set_layers_prototypes_base_on_hidden_rep(self, data, mask, context, fix_protopytes_indices=False):
        X = data.x
        edge_index = data.edge_index
        h = X
        h=h.to(self.device)
        for l,layer in enumerate(self.layers):
            if fix_protopytes_indices:
                if self.protopytes_indices[l] is None:
                    self.protopytes_indices[l]= torch.randint(low=0, high=X[mask].shape[0], size=(self.n_neuron[l],self.n_prototypes))

            layer.set_prototypes(h[mask],self.protopytes_indices[l])
            if context is None:
                h = layer(h, h, edge_index)
            else:
                h = layer(h, context, edge_index)

    def set_layers_prototypes_locally_base_on_hidden_rep(self, data, mask, context, fix_protopytes_indices=False):
        X = data.x
        edge_index = data.edge_index
        h = X
        h = h.to(self.device)
        for l, layer in enumerate(self.layers):
            h_train=h[mask]
            seed_proto = torch.randint(low=0, high=h_train.shape[0],size=[self.n_neuron[l]])
            seed_dists = torch.cdist(h_train[seed_proto],h_train,p=2)
            prototypes_prob_dist = torch.distributions.Categorical(seed_dists)
            protopytes_indices = prototypes_prob_dist.sample([self.n_prototypes]).T

            layer.set_prototypes(h[mask], protopytes_indices)
            if context is None:
                h = layer(h, h, edge_index)
            else:
                h = layer(h, context, edge_index)

    def set_layers_optimizer(self,lr,weight_decay,criterion):
        for layer in self.layers:
            layer.set_opt(lr=lr, weight_decay=weight_decay,criterion=criterion)

    def _set_layer_prototypes_locally_base_on_hidden_rep(self, layer, l, h, mask):


        h_train=h[mask]
        seed_proto = torch.randint(low=0, high=h_train.shape[0],size=[self.n_neuron[l]])
        seed_dists = torch.cdist(h_train[seed_proto],h_train,p=2)
        if torch.sum(seed_dists)>0:
            prototypes_prob_dist = torch.distributions.Categorical(seed_dists)
            protopytes_indices = prototypes_prob_dist.sample([self.n_prototypes]).T
        else:
            protopytes_indices= torch.randint(low=0, high=h_train.shape[0], size=(self.n_neuron[l],self.n_prototypes))
        layer.set_prototypes(h[mask], protopytes_indices)


    def forward(self, data,context):
        X = data.x
        edge_index = data.edge_index

        h=X
        h=h.to(self.device)
        layer_classification =[]
        for layer in self.layers:
            if context is None:
                h = layer(h, h, edge_index)
            else:
                h = layer(h, context, edge_index)
            layer_classification.append(h)

        return layer_classification,h

    def layers_optimization_step(self,data,train_mask, test_mask, valid_mask,context,epochs,test_name="",
                         log_path=".", max_n_epochs_without_improvements=30, early_stopping_threshold=0,
                                 test_epoch=1, set_layer_proto_dynamically=False):

        train_log, test_log, valid_log = prepare_log_files(test_name, log_path)


        X = data.x.to(self.device)
        edge_index = data.edge_index
        y=data.y
        h = X
        h=h.to(self.device)

        print("Network class ", self.network_class," optimization")


        for l,layer in enumerate(self.layers):

            if set_layer_proto_dynamically:
                self._set_layer_prototypes_locally_base_on_hidden_rep(layer=layer,
                                                                      l=l,
                                                                      h=h,
                                                                      mask=train_mask)
            epoch_time_sum = 0
            n_epochs_without_improvements = 0
            best_loss_so_far = -1.0

            layer.train()
            for e in range(epochs):
                epoch_start_time = time.time()

                if context is None:
                    cur_context =h.to(self.device)
                else:
                    cur_context = context.to(self.device)

                loss,h_tm1=layer.opt_step(h, train_mask, cur_context,y, edge_index)

                print("--- layer ",l, " - epoch: ",e," - loss: ",loss.item())

                epoch_time = time.time() - epoch_start_time
                epoch_time_sum += epoch_time

                if e % test_epoch == 0:
                    print("epoch : ", e, " -- loss: ", loss.item())

                    acc_train_set, correct_train_set, n_samples_train_set, loss_train_set = layer.eval_layer(h,edge_index,y,train_mask,cur_context)
                    acc_test_set, correct_test_set, n_samples_test_set, loss_test_set = layer.eval_layer(h,edge_index, y,test_mask,cur_context)
                    acc_valid_set, correct_valid_set, n_samples_valid_set, loss_valid_set = layer.eval_layer(h,edge_index,y,valid_mask, cur_context)
                    print(" -- training acc : ",(acc_train_set, correct_train_set, n_samples_train_set),
                          " -- test_acc : ",(acc_test_set, correct_test_set, n_samples_test_set),
                          " -- valid_acc : ", (acc_valid_set, correct_valid_set, n_samples_valid_set))
                    print("------")

                    train_log.write(
                        "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                            e,
                            l,
                            loss_train_set,
                            acc_train_set,
                            epoch_time_sum / test_epoch,
                            ))

                    train_log.flush()

                    test_log.write(
                        "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                            e,
                            l,
                            loss_test_set,
                            acc_test_set,
                            epoch_time_sum / test_epoch,
                            ))

                    test_log.flush()

                    valid_log.write(
                        "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                            e,
                            l,
                            loss_valid_set,
                            acc_valid_set,
                            epoch_time_sum / test_epoch,
                            ))

                    valid_log.flush()

                    if loss_valid_set < best_loss_so_far or best_loss_so_far == -1:
                        best_loss_so_far = loss_valid_set
                        n_epochs_without_improvements = 0
                        best_epoch = e
                        print("--ES--")
                        print("new_best_model, with loss:", best_loss_so_far)
                        print("------")

                    elif loss_valid_set >= best_loss_so_far + early_stopping_threshold:
                        n_epochs_without_improvements += 1
                    else:
                        n_epochs_without_improvements = 0

                    if n_epochs_without_improvements >= max_n_epochs_without_improvements or e == epochs-1:
                        print("___Early Stopping at epoch ", best_epoch, "____")
                        self.save_model("layer_" + str(l) + "_" + test_name, log_path)
                        break

                    epoch_time_sum = 0
            h = h_tm1

        return h


    def save_model(self,test_name, log_folder='./'):
        torch.save(self.state_dict(), os.path.join(log_folder,test_name+'.pt'))


