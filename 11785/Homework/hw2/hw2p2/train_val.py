import numpy as np
import argparse
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

from utils import init_weights, get_auc, CenterLoss
from model import ResNet, Sphere20

torch.set_num_threads(2)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=256, help='Batch size')
parser.add_argument('--lr',type=float, default=0.01,help='Learning rate')
parser.add_argument('--momentum',type=float,default=0.9,help='momentum')
parser.add_argument('--weight_c',type=float,default=1,help='weight of center loss')
parser.add_argument('--weight_decay',type=float,default=5e-4,help='weightdecay')
parser.add_argument('--num_epoch',type=float,default=100,help='Number of Epoch')
parser.add_argument('--data_path',type=str,default='/home/yihe/Data/hw2/11-785hw2p2-s20/')
parser.add_argument('--save_path',type=str,default='/home/yihe/Data/hw2/model/')
parser.add_argument('--train_folder',type=str,default='train_data/medium')
parser.add_argument('--val_folder',type=str,default='validation_classification/medium')
parser.add_argument('--ver_folder',type=str,default='validation_verification/')
parser.add_argument('--k', type=int, default=12)
args = parser.parse_args()


def train(data_loader, model, criterion_x, criterion_c, optimizer_x, optimizer_c, device, batch_size, weight_c):
    model.train()
    acc = 0.0
    loss = 0.0
    x_loss = 0.0
    c_loss = 0.0
    count = 0

    for figs, target in data_loader:
        figs = figs.to(device)
        target = target.to(device)

        optimizer_c.zero_grad()        
        optimizer_x.zero_grad()
        feature, pred = model(figs)
        

        step_x_loss = criterion_x(pred, target.long())
        step_c_loss = criterion_c(feature,target.long())
        step_c_loss *= weight_c
        step_loss = step_x_loss + step_c_loss
        loss += step_loss.item()
        x_loss += step_x_loss.item()
        c_loss += step_c_loss.item()

        _, pred_labels = torch.max(F.softmax(pred, dim=1),1)
        pred_labels.view(-1)
        step_acc = torch.sum(torch.eq(pred_labels, target)).item()/batch_size
        acc += step_acc
        
        if count%5000 == 4999:
            print(f"Step: {count}, Step Loss: {}, Step Acc: {step_acc}")

        loss.backward()
        optimizer_x.step()
        for param in criterion_c.parameter():
            param.grad.data /= weight_c
        optimizer_c.step()

        count += 1
    
    acc /= count
    loss /= count
    x_loss /= count
    c_loss /= count
    return loss, x_loss, c_loss, acc

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
        step_acc = torch.sum(torch.eq(pred_labels, target)).item()/batch_size
        acc += step_acc
        
        count += 1
    
    acc /= count
    loss /= count

    return loss, acc

# def verification(data_loader, model, device):
#     model.eval()
#     model.change_task('Verification')
#     score = []
#     label = []
#     count = 0
#     for figs, target in data_loader:
#         figs = figs.to(device)
#         target = target.to(device)

#         with torch.no_grad():
#             pred = model(figs)

#         # import pdb;pdb.set_trace()
#         pred = pred.cpu().numpy()
#         if pred.shape[0]!=2:
#             continue
#         feat1 = pred[0,:].reshape(1,-1)
#         feat2 = pred[1,:].reshape(1,-1)
#         value = (cosine_similarity(feat1,feat2)>0.9)
#         truth = target[0]==target[1]
        
#         score.append(value.item())
#         label.append(truth.cpu().item())

#         # if count%50 == 49:
#             # print(count)
#         # print(pred)
#         count += 1
    
#     aus = get_auc(label, score)
#     return aus


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    
    preprocess = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.1),
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

    ver_dataset = ImageFolder(
        root=os.path.join(args.data_path,args.ver_folder),
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

    ver_loader = DataLoader(
        ver_dataset,
        batch_size=2,
        shuffle=True,
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
    model.apply(init_weights)
    # model.load_state_dict(torch.load('30_model1.pth.tar'))
    model.to(device)
    
    criterion_x = nn.CrossEntropyLoss()
    criterion_c = CenterLoss(num_class,feature_dim=2048, device=device)
    optimizer_x = SGD(criterion_x.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum)
    optimizer_c = SGD(criterion_c.parameters(),lr=0.5)
    scheduler = StepLR(optimizer_x, step_size=1, gamma=0.95)

    print('Training begins: ')
    global_acc = 0.0
    for epoch in range(num_epoch):
        epoch += 1
        print (f"Epoch {epoch} starts: ")
        start = time.time()
        train_loss, train_acc = train(train_loader, model, criterion_c, criterion_x, optimizer_c, optimizer_x, device, batch_size, weight_c)
        scheduler.step()
        end = time.time()
        print( f"Epoch {epoch} finishes in {end-start}s; \t Training Loss: {train_loss}; \t  Training Accuracy: {train_acc}")

        
        val_loss, val_acc = val(val_loader, model, criterion_x, device)
        print( f"Validation Loss: {val_loss}; \t Validation Accuracy: {val_acc}")
        # auc = verification(ver_loader,model,device)
        # print( f"Verification Auc: {auc}")
        if global_acc < val_acc:
            torch.save(model.state_dict(), f"./{epoch}_model1.pth.tar")
