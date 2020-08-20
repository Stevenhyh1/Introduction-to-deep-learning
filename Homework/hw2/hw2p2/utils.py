#%%
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from PIL import Image

#%%

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data)

def get_auc(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    return auc


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.bool()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class TestLoader(Dataset):
    def __init__(self, path, transform):
        super(TestLoader,self).__init__()
        self.path = path
        first_img = sorted(os.listdir(self.path))[0]
        self.index_offset = int(os.path.splitext(first_img)[-2])
        self.suffix = os.path.splitext(first_img)[-1]
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(self.path))
    
    def __getitem__(self, index):
        
        img_name = str(self.index_offset+index)+self.suffix
        name = os.path.join(self.path, img_name)
        img = Image.open(name)
        img = self.transform(img)

        return img_name, img

class Verification_Loader(Dataset):
    def __init__(self, data_path, order_file, transform, mode):
        super(Verification_Loader,self).__init__()
        self.data = pd.read_csv(order_file, sep=" ", header=None)
        if mode=='val':
            self.data.columns = ["first", "second", "truth"]
        else:
            self.data.columns = ["first", "second"]
        self.transform = transform
        self.data_path = data_path
        self.mode = mode
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        first_img_name = self.data['first'][index]
        first_img = Image.open(os.path.join(self.data_path, first_img_name))
        first_img = self.transform(first_img)
        second_img_name = self.data['second'][index]
        second_img = Image.open(os.path.join(self.data_path, second_img_name))
        second_img = self.transform(second_img)
        if self.mode == 'val':
            return first_img, second_img, first_img_name, second_img_name, self.data['truth'][index]

        return first_img, second_img, first_img_name, second_img_name


def index_mapping(path='/home/yihe/Data/hw2/11-785hw2p2-s20/train_data/medium'):
    count = 0
    dict = {}
    for folder_name in sorted(os.listdir(path)):
        dict[count] = int(folder_name)
        count += 1
    return dict
        


if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    if len(args) < 1:
        exit()
    score_csv = args[0]
    label_csv = args[1]
    y_score = pd.read_csv(score_csv).score.values
    y_true = pd.read_csv(label_csv).score.values
    auc = get_auc(y_true, y_score)
    print ("AUC: ", auc)