from sklearn.model_selection import train_test_split
import torch

def get_masks(n_sample, test_valid_perc, rnd_seed=0):
    data_indices=range(n_sample)
    idx_train, idx_test = train_test_split(data_indices, test_size=test_valid_perc, random_state=rnd_seed)  #
    idx_train, idx_val = train_test_split(idx_train, test_size=len(idx_test), random_state=rnd_seed)
    train_mask=torch.zeros(n_sample).long()
    test_mask = torch.zeros(n_sample).long()
    val_mask = torch.zeros(n_sample).long()
    return train_mask.scatter_(0,torch.LongTensor(idx_train),1).bool(),test_mask.scatter_(0,torch.LongTensor(idx_test),1).bool(),val_mask.scatter_(0,torch.LongTensor(idx_val),1).bool()


