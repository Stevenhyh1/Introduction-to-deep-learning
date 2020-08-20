import numpy as np
import argparse
import os
import time

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

def train(data_loader, model, criterion, optimizer, device, batch_size, center_loss=None, angle_loss=None, gamma=0):
    model.train()
    acc = 0.0
    loss = 0.0
    count = 0

    for figs, target in data_loader:
        
        figs = figs.to(device)
        target = target.to(device)
       
        optimizer.zero_grad()
        if center_loss is not None:
            feats, pred = model(figs)
            c_loss = center_loss(feats, target)
            x_loss = criterion(pred, target.long())
            step_loss = c_loss * gamma + x_loss
        elif angle_loss is not None:
            pred = model(figs)
            step_loss = angle_loss(pred, target.long())
        else:
            pred = model(figs)
            step_loss = criterion(pred, target.long())
        

        step_loss.backward()
        optimizer.step()

        _, pred_labels = torch.max(F.softmax(pred, dim=1),1)
        pred_labels = pred_labels.view(-1)
        step_acc = torch.sum(torch.eq(pred_labels, target)).item()/batch_size
        acc += step_acc
        loss += step_loss.item()

        if count%500 == 499:
            if center_loss is not None:
                print(f"Step: {count}, Step Loss: {step_loss}, Step Acc: {step_acc}, Center Loss: {c_loss}, X Loss: {x_loss}")
            else:
                print(f"Step: {count}, Step Loss: {step_loss}, Step Acc: {step_acc}")

        count += 1
    
    acc /= count
    loss /= count

    return loss, acc

def val(data_loader, model, criterion, device, center_loss=None):
    model.eval()
    acc = 0.0
    loss = 0.0
    count = 0

    for figs, target in data_loader:

        figs = figs.to(device)
        target = target.to(device)

        with torch.no_grad():
            if center_loss is not None:
                _, pred = model(figs)
            else:
                pred = model(figs)

        step_loss = criterion(pred, target.long())
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int, default=256, help='Batch size')
    parser.add_argument('--lr',type=float, default=0.01,help='Learning rate')
    parser.add_argument('--momentum',type=float,default=0.9,help='momentum')
    parser.add_argument('--weight_c',type=float,default=1,help='weight of center loss')
    parser.add_argument('--weight_decay',type=float,default=5e-4,help='weightdecay')
    parser.add_argument('--num_epoch',type=float,default=40,help='Number of Epoch')
    parser.add_argument('--data_path',type=str,default='/home/yihe/Data/hw2p2/')
    parser.add_argument('--mode',type=str,default='pretrain', choices=['pretrain', 'finetune', 'cont_pre', 'cont_tune', 'eval'], help='Pretrain mode or Finetune Mode')
    parser.add_argument('--loss',type=str,default='None',choices=['center','angle','None'], help='Loss of the model')
    args = parser.parse_args()

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

    if args.mode == 'pretrain' or 'cont_pre':
        train_folder = 'train_data/medium'
        val_folder = 'validation_classification/medium'
    elif args.mode == 'finetune':
        train_folder = 'train_data/large'
        val_folder = 'validation_classification/large'
    
    train_dataset = ImageFolder(
        root=os.path.join(args.data_path,train_folder),
        transform=preprocess
    )

    val_dataset = ImageFolder(
        root=os.path.join(args.data_path,val_folder),
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
    if args.loss == 'center':
        model.change_task('Both')

    if args.mode == 'finetune':
        pretrain_dict = torch.load('models/65_model.pth.tar')
        model_dict = model.state_dict()
        update_dict = {k: v for k, v in pretrain_dict.items() if k != 'module.fc.weight'}
        model_dict.update(update_dict)
    elif args.mode == 'cont_tune':
        model.load_state_dict(torch.load('models/19_model_finetune.pth.tar'))
    elif args.mode == 'cont_pre':
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load('models/29_model_cont_pre.pth.tar'))
    model.to(device)
    # import pdb; pdb.set_trace()
    
    center_loss = None
    angle_loss = None
    gamma = 0
    if args.loss == 'center':
        center_loss = CenterLoss(num_class, 2048)
        gamma = 0.001
    elif args.loss == 'angle':
        angle_loss = AngleLoss()
    elif args.loss != 'None':
        raise Exception(f"Loss {args.loss} is not supported")
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    print('Training begins: ')
    global_acc = 0.0

    for epoch in range(num_epoch):
        epoch += 30
        if args.mode != eval:
            print (f"Epoch {epoch} starts: ")
            print( f"Gamma: {gamma}")
            start = time.time()
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, device, batch_size, center_loss, angle_loss, gamma)
            scheduler.step()
            end = time.time()
            print( f"Epoch {epoch} finishes in {end-start}s; \t Training Loss: {train_loss}; \t  Training Accuracy: {train_acc}")

        
        val_loss, val_acc = val(val_loader, model, criterion, device, center_loss)
        print( f"Validation Loss: {val_loss}; \t Validation Accuracy: {val_acc}")
        if global_acc < val_acc:
            torch.save(model.state_dict(), f"models/{epoch}_model_{args.mode}.pth.tar")
            global_acc = val_acc
        if args.loss == 'center':
            gamma *= 1
