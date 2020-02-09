import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def load_data (file_path, k):
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

def load_label (file_path):
    label = np.load(file_path, allow_pickle=True)
    label_array = np.concatenate(label)
    return label_array


class SpeechDataset(Dataset):
    """Pytorch map-style dataset"""
    def __init__(self, data, label, data_index, k):
        self.data = data
        self.label = label
        self.data_index = data_index
        self.k = k
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        k=self.k
        utt_idx = np.searchsorted(self.data_index,idx,side='right')-1
        utt = self.data[self.data_index[utt_idx]]
        local_idx = idx-self.data_index[utt_idx]
        local_len = len(utt)
        if local_idx<k:
            frame_data = np.hstack((np.zeros(40*(k-local_idx)),utt[:local_idx+k+1].flatten()))
        elif local_idx+k >= local_len:
            frame_data = np.hstack((utt[local_idx-k:].flatten(), np.zeros(40*(local_idx+k-local_len+1))))
        else:
            frame_data = utt[local_idx-k:local_idx+k+1].flatten()
        return torch.tensor(frame_data), torch.tensor(self.label[idx])