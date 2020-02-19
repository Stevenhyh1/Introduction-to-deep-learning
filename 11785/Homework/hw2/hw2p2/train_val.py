import numpy as np
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import faulthandler

faulthandler.enable() 

import utils
from model import CNN

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=512, help='Batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='Learning rate')
parser.add_argument('--momentum',type=float,default=0.9,help='SGD momentum')
parser.add_argument('--num_epoch',type=float,default=15,help='Number of Epoch')
parser.add_argument('--data_path',type=str,default='/home/yihe/Data/hw2/11-785hw2p2-s20/')
parser.add_argument('--train_folder',type=str,default='train_data/')
parser.add_argument('--val_folder',type=str,default='validation_classification/')
parser.add_argument('--k', type=int, default=12)
args = parser.parse_args()

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(
        root=os.path.join(args.data_path,args.train_folder),
        target_transform=preprocess
    )

    val_dataset = ImageFolder(
        root=os.path.join(args.data_path,args.val_folder),
        target_transform=preprocess
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    model = 


