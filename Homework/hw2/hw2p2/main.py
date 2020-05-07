import numpy as np
import argparse
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import faulthandler
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

faulthandler.enable() 

from utils import init_weights, get_auc, CenterLoss, AngleLoss
from model import ResNet
# from sphere import ResNet

torch.set_num_threads(4)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=256, help='Batch size')
parser.add_argument('--lr',type=float, default=0.01,help='Learning rate')
parser.add_argument('--momentum',type=float,default=0.9,help='momentum')
parser.add_argument('--weight_c',type=float,default=1,help='weight of center loss')
parser.add_argument('--weight_decay',type=float,default=5e-4,help='weightdecay')
parser.add_argument('--num_epoch',type=float,default=100,help='Number of Epoch')
parser.add_argument('--data_path',type=str,default='/home/yihe/Data/hw2/11-785hw2p2-s20/')
parser.add_argument('--save_path',type=str,default='/home/yihe/Data/hw2/model/')
parser.add_argument('--train_folder',type=str,default='train_data/large')
parser.add_argument('--val_folder',type=str,default='validation_classification/large')
parser.add_argument('--ver_folder',type=str,default='validation_verification/')
parser.add_argument('--k', type=int, default=12)
args = parser.parse_args()


def train(data_loader, model, criterion, optimizer, device, batch_size):
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

        step_loss.backward()
        optimizer.step()

        _, pred_labels = torch.max(F.softmax(pred, dim=1),1)
        pred_labels = pred_labels.view(-1)
        step_acc = torch.sum(torch.eq(pred_labels, target)).item()/batch_size
        acc += step_acc
        loss += step_loss.item()

        if count%1000 == 999:
            print(f"Step: {count}, Step Loss: {step_loss}, Step Acc: {step_acc}")

        count += 1
    
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

        #Shape of predition: ((B, C), (B, C))
        #Shape of target: (B)
        # import pdb; pdb.set_trace()

        step_loss = criterion(pred, target.long())
        # step_lossd = step_loss.data[0]
        loss += step_loss.item()
        
        _, pred_labels = torch.max(F.softmax(pred, dim=1),1)
        pred_labels.view(-1)
        step_acc = torch.sum(torch.eq(pred_labels, target)).item()/batch_size
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
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomCrop(size=28, pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_process = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(
        root=os.path.join(args.data_path,args.train_folder),
        transform=preprocess
    )

    val_dataset = ImageFolder(
        root=os.path.join(args.data_path,args.val_folder),
        transform=val_process
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
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    hidden_sizes = [3,4,6,3]
    momentum = args.momentum
    weight_c = args.weight_c

    model = ResNet(num_channel,num_class,hidden_sizes)
    model.load_state_dict(torch.load('54_model1.pth.tar'))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    print('Training begins: ')
    global_acc = 0.0
    for epoch in range(num_epoch):
        epoch += 61
        print (f"Epoch {epoch} starts: ")
        start = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device, batch_size)
        scheduler.step()
        end = time.time()
        print( f"Epoch {epoch} finishes in {end-start}s; \t Training Loss: {train_loss}; \t  Training Accuracy: {train_acc}")

        
        val_loss, val_acc = val(val_loader, model, criterion, device)
        print( f"Validation Loss: {val_loss}; \t Validation Accuracy: {val_acc}")
        if global_acc < val_acc:
            torch.save(model.state_dict(), f"./{epoch}_model1.pth.tar")
