import numpy as np
import torch
import argparse

from model import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=32, help='Batch size')
parser.add_argument('--lr',type=float,default=0.001,help='Learning rate')
parser.add_argument('--momentum',type=float,default=0.9,help='SGD momentum')
parser.add_argument('--num_epoch',type=float,default=50,help='Number of Epoch')
parser.add_argument('--data_path',type=str,default='./dev.npy')
parser.add_argument('--label_path',type=str,default='./dev_labels.npy')

if __name__ == '__main__':

    #GPU Check
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    