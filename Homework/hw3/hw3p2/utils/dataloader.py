import torch
import os
import numpy as np
from torch.utils.data import Dataset

def load_data (file_path):
    data = np.load(file_path, allow_pickle=True)
    data_array = []
    eps = 1e-8
    for i in range(len(data)):
        cur_data = data[i]
        mean = np.mean(cur_data, axis=1).reshape(-1,1)
        var = np.var(cur_data, axis=1).reshape(-1,1)
        data_array.append((cur_data-mean)/np.sqrt(var+eps))
    return data_array

def load_label (file_path):
    label = np.load(file_path, allow_pickle=True)
    label = label+1
    return list(label)

class SpeechDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = list(data)
        self.label = label
        if label != None:
            self.label = list(label)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label == None:
            return torch.tensor(self.data[idx])
        return torch.tensor(self.data[idx]), torch.tensor(self.label[idx])

if __name__ == "__main__":
    data_path = '/home/yihe/Data/hw3p2/wsj0_dev.npy'
    data_array = load_data(data_path)
    

    label_path = '/home/yihe/Data/hw3p2/wsj0_dev_merged_labels.npy'
    label_array = load_label(label_path)

    # import pdb; pdb.set_trace()
