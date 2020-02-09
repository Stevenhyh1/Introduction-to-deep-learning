import numpy as np
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import utils
from model import MLP

os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=256, help='Batch size')
parser.add_argument('--lr',type=float,default=0.0002,help='Learning rate')
parser.add_argument('--momentum',type=float,default=0.9,help='SGD momentum')
parser.add_argument('--num_epoch',type=float,default=15,help='Number of Epoch')
parser.add_argument('--data_path',type=str,default='/home/yihe/Data')
parser.add_argument('--label_path',type=str,default='/home/yihe/Data')
parser.add_argument('--k', type=int, default=10)
args = parser.parse_args()

def Train(data, model, criterion, optimizer):
    # loss_list = []
    # acc_list = []
    epoch_loss = 0
    epoch_acc = 0
    count = 0
    model.train()
    print(f'Training: ')
    for frame, label in data:
        
        frame = frame.to(device)
        label = label.to(device)    
        optimizer.zero_grad()
        pred = model(frame.float())
        loss = criterion(pred, label)
        _, indices = pred.max(1)
        count += 1
        epoch_acc += np.mean((indices==label).int().cpu().numpy())
        epoch_loss += loss.detach()
        # loss_list.append(loss)
        # acc_list.append(acc)
        loss.backward()
        optimizer.step()

    return epoch_loss/count, epoch_acc/count

def Val(data, model, criterion):
    print(f'Validation: ')
    # loss_list = []
    # acc_list = []
    epoch_loss = 0
    epoch_acc = 0
    count = 0
    model.eval()
    for frame, label in data:
        
        frame = frame.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = model(frame.float())
        loss = criterion(pred, label)
        _, indices = pred.max(1)
        acc = np.mean((indices==label).int().cpu().numpy())
        count += 1
        epoch_loss += loss.detach()
        epoch_acc += acc

    return epoch_loss/count, epoch_acc/count

if __name__ == '__main__':

    '''Step 0: Preprocessing'''
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

    '''Step 1: Load Dataset'''
    print('Loading Training Data...')
    train_data, train_idx = utils.load_data(os.path.join(args.data_path,'train.npy'), k)
    train_label = utils.load_label(os.path.join(args.label_path,'train_labels.npy'))
    train_dataset = utils.SpeechDataset(train_data,train_label,train_idx,k)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    print('Loading Validation Data...')
    val_data, val_idx = utils.load_data(os.path.join(args.data_path, 'dev.npy'), k)
    val_label = utils.load_label(os.path.join(args.label_path,'dev_labels.npy'))
    val_dataset = utils.SpeechDataset(val_data,val_label,val_idx,k)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    print('Data Loaded!')

    '''Step 2: Model Initialization'''
    #Model
    model = MLP(input_dim, output_dim)
    model.to(device)

    #Loss Function
    criterion = nn.CrossEntropyLoss()

    #Optimizer
    optimizer = Adam(model.parameters(),lr=lr)

    '''Step 3: Train the Model'''
    print('Training begins: ')

    global_acc = 0
    for epoch in range(num_epoch):
        print(f'Epoch {epoch} starts:')
        train_start = time.time()
        train_loss, train_acc = Train(
            train_dataloader,
            model,
            criterion,
            optimizer
        )
        train_end = time.time()

        print(f"Epoch {epoch} completed in: {train_end-train_start}s \t Loss: {train_loss} \t Acc: {train_acc}")

        val_start = time.time()
        val_loss, val_acc = Val(
            val_dataloader,
            model,
            criterion
        )
        val_end = time.time()
        if val_acc > global_acc:
            torch.save(model.state_dict(), f"./{epoch}_model.pth.tar")
            global_acc = val_acc
        print(f"Validation Loss: {val_loss} \t Validation Acc: {val_acc}")