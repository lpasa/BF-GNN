import os
import datetime
import torch



def printParOnFile(test_name, log_dir, par_list):

    assert isinstance(par_list, dict), "par_list as to be a dictionary"
    f=open(os.path.join(log_dir,test_name+".log"),'w+')
    f.write(test_name)
    f.write("\n")
    f.write(str(datetime.datetime.now().utcnow()))
    f.write("\n\n")
    for key, value in par_list.items():
        f.write(str(key)+": \t"+str(value))
        f.write("\n")

def get_SP_from_batch(batch):
    index = batch.sp_index
    values = batch.sp_v[:, 0]
    shape = batch.sp_shape
    shape = torch.reshape(shape,(int(shape.shape[0]/2),2))
    shape = list(torch.sum(shape, dim=0))
    SP_mat = torch.sparse_coo_tensor(index, values,shape)
    return SP_mat.to_dense()

def get_k_SP(m,k):
    return ((m==k).float())

def prepare_log_files(test_name, log_dir):
    train_log = open(os.path.join(log_dir, (test_name + "_train")), 'w+')
    test_log = open(os.path.join(log_dir, (test_name + "_test")), 'w+')
    valid_log = open(os.path.join(log_dir, (test_name + "_valid")), 'w+')

    for f in (train_log, test_log, valid_log):
        f.write("test_name: %s \n" % test_name)
        f.write(str(datetime.datetime.now()) + '\n')
        f.write("#epoch \t layer \t loss \t acc \t avg_epoch_time \n")

    return train_log, test_log, valid_log