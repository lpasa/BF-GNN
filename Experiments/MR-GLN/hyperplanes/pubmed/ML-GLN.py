import os
import os.path as osp
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

import torch
torch.set_printoptions(threshold=5000)

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from data_reader.LoadCitation import get_masks
from model.MRGLN import MRGLN
from utils.utils_method import printParOnFile

if __name__ == '__main__':
    n_epochs = 200
    k_list = [1, 2, 4, 6, 8, 16]
    l_list=[1, 2, 3, 4, 5, 6, 7, 8]
    lr_list = [0.1,0.2,0.01,0.001]
    diffusion_op_list = ['A', 'L']
    weight_decay_list = [0,5e-3,5e-4]

    test_epoch = 1
    n_run = 5
    max_n_epochs_without_improvements = 15

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #DATASET
    dataset_name = 'Pubmed'
    path = osp.join('~/Dataset/', dataset_name)
    dataset = Planetoid(path, dataset_name, split='full', transform=T.TargetIndegree())
    data = dataset[0].to(device)
    test_type = "MR-GLN"


    for run in range(n_run):
        for weight_decay in weight_decay_list:
            for l in l_list:
                for lr in lr_list:
                    for n_prototypes in k_list:
                        for diffusion_op in diffusion_op_list:




                            test_name = "run_" + str(run) +"_"+test_type

                            test_name = test_name + "_data-" + dataset_name +\
                                        "_lr-" + str(lr) \
                                        + "_weight-decay-" + str(weight_decay) +\
                                        "_k-" + str(l) \
                                        + "_diffusion_op-" + str(diffusion_op)  \
                                        + "_n_context-" + str(n_prototypes)
                            print(test_name)
                            training_log_dir = os.path.join("./test_log/"+test_type, test_name)
                            if not os.path.exists(training_log_dir):
                                os.makedirs(training_log_dir)

                                printParOnFile(test_name=test_name, log_dir=training_log_dir, par_list={"dataset_name": dataset_name,
                                                                                                        "learning_rate": lr,
                                                                                                        "weight_decay": weight_decay,
                                                                                                        "l": l,
                                                                                                        "diffusion_op": diffusion_op,
                                                                                                        "test_epoch": test_epoch,
                                                                                                        })


                                train_mask, test_mask,val_mask = get_masks(data.x.shape[0],0.2,run+1)

                                model = MRGLN(in_channels=data.x.shape[1],
                                              n_class=max(data.y)+1,
                                              k=l,
                                              diffusion_op=diffusion_op,
                                              n_proto=n_prototypes,
                                              GLN_type='hyper',
                                              device=device).to(device)


                                model.set_networks_optimizer(lr=lr,weight_decay=weight_decay,criterion=torch.nn.BCELoss())


                                model.PLGFNN_optimization_and_test(data=data,
                                                                    train_mask=train_mask,
                                                                    test_mask=test_mask,
                                                                    valid_mask=val_mask,
                                                                    context=data.x,
                                                                    epochs=n_epochs,
                                                                    test_name=test_name,
                                                                    log_path=training_log_dir,
                                                                    max_n_epochs_without_improvements=max_n_epochs_without_improvements)


                                if str(device) == 'cuda':
                                    del model
                                    torch.cuda.empty_cache()

                            else:
                                print("test has been already execute")


