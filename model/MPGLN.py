import os
import datetime
import torch
from model.class_network import MPClassNetwork



class MPGLN(torch.nn.Module):
    def __init__(self, in_channels,n_class, n_neuron,context_dim, n_prototypes, n_layer, conv=None, device=None, input_as_context = False):
        super(MPGLN, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels = in_channels
        self.n_neuron = n_neuron
        self.n_prototypes = n_prototypes
        self.context_dim = context_dim
        self.n_layer = n_layer

        self.class_networks = torch.nn.ModuleList()
        for c in range(n_class):
            self.class_networks.append(MPClassNetwork(in_channels=in_channels,
                                                      n_neuron=n_neuron,
                                                      context_dim=context_dim,
                                                      n_prototypes=n_prototypes,
                                                      network_class=c,
                                                      n_layer=n_layer,
                                                      conv=conv,
                                                      device=device,
                                                      input_as_context=input_as_context))
        self.out_fun=torch.nn.Sigmoid()


    def set_networks_layer_protopytes_based_on_h(self, data, mask, context):
        for class_net in self.class_networks:
            class_net.set_layers_prototypes_base_on_hidden_rep(data, mask, context, fix_protopytes_indices=False)

    def set_networks_layer_prototypes_localized_base_on_hidden(self, data, mask, context):
        for class_net in self.class_networks:
            class_net.set_layers_prototypes_locally_base_on_hidden_rep(data, mask, context, fix_protopytes_indices=False)

    def set_networks_layer_protopytes_based_on_x(self, data, mask):
        for class_net in self.class_networks:
            class_net.set_layers_prototypes_base_on_input_rep(data, mask, fix_protopytes_indices=False)


    def set_networks_optimizer(self,lr,weight_decay,criterion):
        for class_net in self.class_networks:
            class_net.set_layers_optimizer(lr,weight_decay,criterion)

    def train_and_test(self, data, train_mask, test_mask, valid_mask,epochs,context=None,test_name="",
                log_path=".", max_n_epochs_without_improvements=30, early_stopping_threshold=0, test_epoch=1,
                       set_layer_proto_dynamically=False):
        h_last_layers=[]
        for c,class_net in enumerate(self.class_networks):

            h_c = class_net.layers_optimization_step(data=data,
                                               train_mask=train_mask,
                                               test_mask=test_mask,
                                               valid_mask=valid_mask,
                                               context=context,
                                               epochs=epochs,
                                               test_name= "class_"+str(c)+"_"+test_name,
                                               log_path=log_path,
                                               max_n_epochs_without_improvements=max_n_epochs_without_improvements,
                                               early_stopping_threshold=early_stopping_threshold,
                                               test_epoch=test_epoch,
                                               set_layer_proto_dynamically=set_layer_proto_dynamically)
            h_last_layers.append(h_c)

        #concat the output of the last layer of each network
        prob_classes=self.out_fun(torch.cat(h_last_layers,dim=1))

        #eval model
        acc_train,_,_ = self.eval_model(prob_classes,data.y,train_mask)
        acc_test,_,_ = self.eval_model(prob_classes,data.y,test_mask)
        acc_valid,_,_ = self.eval_model(prob_classes,data.y,valid_mask)


        results_file = open(os.path.join(log_path, (test_name + "_results")), 'w+')
        results_file.write("test_name: %s \n" % test_name)
        results_file.write(str(datetime.datetime.now()) + '\n')
        results_file.write("acc training \t  acc test \t acc valid \n")
        results_file.write("{: .8f}\t{: .8f}\t{: .8f}\n".format(acc_train,acc_test,acc_valid))

        print(" acc training:{: .8f}\n acc test: {: .8f}\n acc valid: {: .8f}\n".format(acc_train,acc_test,acc_valid))
        print("------------------------")


    def eval_model(self, prob,y, mask):
        pred = prob.max(1)[1]
        n_samples = len(y[mask])
        correct = pred[mask].eq(y[mask]).sum().item()
        acc = correct / n_samples
        return  acc,correct, n_samples

    def free_mem(self):
        if str(self.device) == 'cuda':
            for net in self.class_networks:
                del net
            torch.cuda.empty_cache()

