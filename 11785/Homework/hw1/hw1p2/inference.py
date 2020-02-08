import numpy as np
import pandas as pd
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import utils
from model import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=256, help='Batch size')
parser.add_argument('--lr',type=float,default=0.001,help='Learning rate')
parser.add_argument('--momentum',type=float,default=0.9,help='SGD momentum')
parser.add_argument('--num_epoch',type=float,default=15,help='Number of Epoch')
parser.add_argument('--data_path',type=str,default='/home/yihe/Data')
parser.add_argument('--label_path',type=str,default='/home/yihe/Data')
parser.add_argument('--save_file',type=str,default='./result.csv')
parser.add_argument('--model_file',type=str,default='./14_model.pth.tar')
parser.add_argument('--k', type=int, default=12)
args = parser.parse_args()

def Val(data, model):

    pred_list = []
    for frame, label in data:
        
        frame = frame.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = model(frame.float())
        _, indices = pred.max(1)
        indices = indices.int().cpu().numpy().reshape(-1,1)
        cur_pred = np.hstack((label.int().cpu().numpy().reshape(-1,1),indices))
        pred_list.append(cur_pred)

    return np.vstack(pred_list).astype(int)

if __name__ == '__main__':

    #GPU Check
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #Hyperparameters
    batch_size = args.batch_size
    lr = args.lr
    momentum = args.momentum
    num_epoch = args.num_epoch
    data_path = args.data_path
    label_path = args.label_path
    k = args.k
    input_dim = 40*(2*k+1)
    output_dim = 138

    print('Loading Inference Data...')
    val_data, val_idx = utils.load_data(os.path.join(args.data_path, 'test.npy'), k)
    val_label = np.arange(val_idx[-1]+len(val_data[val_idx[-1]]))
    val_dataset = utils.SpeechDataset(val_data,val_label,val_idx,k)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print('Data Loaded!')

    #Load Model
    model = MLP(input_dim, output_dim)
    model.to(device)
    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    print('Inference begins: ')

    pred_result = Val(
        val_dataloader,
        model
    )

    datafile = pd.DataFrame({'id': pred_result[:,0], 'label': pred_result[:,1]}, index=None)
    datafile.to_csv('result.csv', index=False)
