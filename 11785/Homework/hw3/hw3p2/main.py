
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset


torch.set_num_threads(4)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=256, help='Batch size')
parser.add_argument('--lr',type=float, default=0.01,help='Learning rate')
parser.add_argument('--momentum',type=float,default=0.9,help='momentum')
parser.add_argument('--weight_c',type=float,default=1,help='weight of center loss')
parser.add_argument('--weight_decay',type=float,default=5e-4,help='weightdecay')
parser.add_argument('--num_epoch',type=float,default=100,help='Number of Epoch')
parser.add_argument('--k', type=int, default=12)
args = parser.parse_args()

if __name__ == "__main__":
    
    if torch.cuda.is_availabel():
        device = 'cuda'
    else:
        device = 'cpu'

    