import torch
import os
import numpy as np
from torch.utils.data import Dataset

def load_data (file_path):
    data = np.load(file_path, allow_pickle=True)
    data_dict = {}
    cur = 0
    eps = 1e-8
    index_array = np.zeros(len(data),dtype=int)
    for i in range(len(data)):
        cur_data = data[i]
        mean = np.mean(cur_data, axis=1).reshape(-1,1)
        var = np.var(cur_data,axis=1).reshape(-1,1)
        data_dict[cur] = (cur_data-mean)/np.sqrt(var+eps)
        index_array[i] = cur
        cur += len(cur_data)
    return data_dict, index_array

if __name__ == "__main__":
    data_path = '/home/stevenhyh/Data/hw3p2/wsj0_train'
    data_dict, index_array = load_data(data_path)
    import pdb; pdb.set_trace()