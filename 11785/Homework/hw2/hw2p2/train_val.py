import numpy as np
import argparse
import os
import time
import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import faulthandler

faulthandler.enable() 

import utils
from model import CNN

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=512, help='Batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='Learning rate')
parser.add_argument('--momentum',type=float,default=0.9,help='SGD momentum')
parser.add_argument('--num_epoch',type=float,default=15,help='Number of Epoch')
parser.add_argument('--data_path',type=str,default='/home/yihe/Data')
parser.add_argument('--label_path',type=str,default='/home/yihe/Data')
parser.add_argument('--k', type=int, default=12)
args = parser.parse_args()