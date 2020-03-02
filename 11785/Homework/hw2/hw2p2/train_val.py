import numpy as np
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import faulthandler
from PIL import Image

faulthandler.enable() 

from utils import init_weights
from model import ResNet

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=32, help='Batch size')
parser.add_argument('--lr',type=float, default=0.001,help='Learning rate')
parser.add_argument('--momentum',type=float,default=0.9,help='momentum')
parser.add_argument('--weight_decay',type=float,default=5e-5,help='weightdecay')
parser.add_argument('--num_epoch',type=float,default=15,help='Number of Epoch')
parser.add_argument('--data_path',type=str,default='/home/yihe/Data/hw2/11-785hw2p2-s20/')
parser.add_argument('--save_path',type=str,default='/home/yihe/Data/hw2/model/')
parser.add_argument('--train_folder',type=str,default='train_data/medium')
parser.add_argument('--val_folder',type=str,default='validation_classification/medium')
parser.add_argument('--k', type=int, default=12)
args = parser.parse_args()


def train(data_loader, model, criterion, optimizer, device):
    model.train()
    acc = 0.0
    loss = 0.0
    count = 0
    for figs, target in data_loader:
        figs = figs.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(figs)
        
        step_loss = criterion(pred, target.long())
        loss += step_loss.item()
        
        import pdb; pdb.set_trace()

        _, pred_labels = torch.max(F.softmax(pred, dim=1),1)
        pred_labels.view(-1)
        step_acc = torch.sum(torch.eq(pred_labels, target)).item()
        acc += step_acc
        
        step_loss.backward()
        count += 1

        if count % 50 ==49:
            print(f"Step {count}: Loss: {step_loss}")
    
    acc /= count
    loss /= count
    return loss, acc

def val(data_loader, model, criterion, device):
    model.eval()
    acc = 0.0
    loss = 0.0
    count = 0
    for figs, target in data_loader:
        figs = figs.to(device)
        target = target.to(device)

        with torch.no_grad():
            pred = model(figs)

        step_loss = criterion(pred, target.long())
        loss += step_loss.item()
        
        _, pred_labels = torch.max(F.softmax(pred, dim=1),1)
        pred_labels.view(-1)
        step_acc = torch.sum(torch.eq(pred_labels, target)).item()
        acc += step_acc
        
        count += 1
    
    acc /= count
    loss /= count

    return loss, acc

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
        transform=preprocess
    )

    val_dataset = ImageFolder(
        root=os.path.join(args.data_path,args.val_folder),
        transform=preprocess
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1
    )


    #Hyperparameters
    num_epoch = args.num_epoch
    num_channel = 3
    num_class = len(train_dataset.classes)
    lr = args.lr
    weight_decay = args.weight_decay
    hidden_sizes = [3,4,6,3]

    model = ResNet(num_channel,num_class,hidden_sizes)
    model.apply(init_weights)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print('Training begins: ')
    global_acc = 0.0
    for epoch in range(num_epoch):
        epoch += 1
        print (f"Epoch {epoch} starts: ")
        start = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        end = time.time()
        print( f"Epoch {epoch} finishes in {end-start}s; \t Training Loss: {train_loss}; \t  Training Accuracy: {train_acc}")

        
        val_loss, val_acc = val(val_loader, model, criterion, device)
        print( f"Validation Loss: {val_loss}; \t Validation Accuracy: {val_acc}")
        if global_acc < val_acc:
            torch.save(model.state_dict(), args.save_path)
