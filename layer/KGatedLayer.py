import torch
torch.set_printoptions(profile="full")
from layer.layer_class import MR_ClassLayer
from utils.utils_method import prepare_log_files
import time
import os



class KGateLayer(torch.nn.Module):
    def __init__(self, in_channels, n_class=2, n_proto=2,GLN_type="proto", context_dim=0, device=None):
        super(KGateLayer, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels = in_channels
        self.n_class = n_class
        self.n_prototypes=n_proto

        self.layer= torch.nn.ModuleList()
        if context_dim == 0:
            context_dim = in_channels

        for c_i in range(n_class):
            self.layer.append(MR_ClassLayer(in_channels= in_channels,
                                            n_neuron=1,
                                            context_dim=context_dim,
                                            n_prototypes=n_proto,
                                            layer_class=c_i,
                                            conv_type=GLN_type,
                                            device=self.device))
        self.reset_parameters()

    def reset_parameters(self):
        pass



    def set_networks_layer_protopytes_based_on_x(self, data, mask):
        for neuron in self.layer:
            neuron.set_prototypes(data.x, torch.randint(low=0, high=data.x[mask].shape[0], size=(1,neuron.n_prototypes)))

    def set_layer_optimizer(self,lr,weight_decay,criterion):
        for neuron in self.layer:
            neuron.set_opt(lr=lr, weight_decay=weight_decay,criterion=criterion)


    def forward(self,X,context):
        H =[]
        for neuron in self.layer:
            H.append(neuron(X,context))
        return H#retrun list, for perfromn cat only ones by the net

    def gated_layer_optimization_step(self, X,y, train_mask, test_mask, valid_mask, context, epochs, test_name="",
                                 log_path=".", max_n_epochs_without_improvements=30, early_stopping_threshold=0,
                                 test_epoch=1):

        train_log, test_log, valid_log = prepare_log_files(test_name, log_path)


        for l, neuron in enumerate(self.layer):
            print("Gated Layer class ", neuron.layer_class, " optimization")
            epoch_time_sum = 0
            n_epochs_without_improvements = 0
            best_loss_so_far = -1.0

            neuron.train()
            for e in range(epochs):
                epoch_start_time = time.time()

                if context is None:
                    cur_context = X.to(self.device)
                else:
                    cur_context = context.to(self.device)

                loss, h = neuron.opt_step(X, train_mask, cur_context, y)


                print("--- layer ", l, " - epoch: ", e, " - loss: ", loss.item())

                epoch_time = time.time() - epoch_start_time
                epoch_time_sum += epoch_time

                if e % test_epoch == 0:
                    print("epoch : ", e, " -- loss: ", loss.item())

                    acc_train_set, correct_train_set, n_samples_train_set, loss_train_set = neuron.eval_layer(X,
                                                                                                             y,
                                                                                                             train_mask,
                                                                                                             cur_context)
                    acc_test_set, correct_test_set, n_samples_test_set, loss_test_set = neuron.eval_layer(X,
                                                                                                         y, test_mask,
                                                                                                         cur_context)
                    acc_valid_set, correct_valid_set, n_samples_valid_set, loss_valid_set = neuron.eval_layer(X,y,
                                                                                                             valid_mask,
                                                                                                             cur_context)
                    print(" -- training acc : ", (acc_train_set, correct_train_set, n_samples_train_set),
                          " -- test_acc : ", (acc_test_set, correct_test_set, n_samples_test_set),
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

                    if n_epochs_without_improvements >= max_n_epochs_without_improvements or e == epochs - 1:
                        print("___Early Stopping at epoch ", best_epoch, "____")
                        # self.save_model("layer_" + str(l) + "_" + test_name, log_path)
                        break

                    epoch_time_sum = 0



    def save_model(self,test_name, log_folder='./'):
        torch.save(self.state_dict(), os.path.join(log_folder,test_name+'.pt'))



