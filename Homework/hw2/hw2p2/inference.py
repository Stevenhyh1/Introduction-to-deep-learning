import numpy as np
import argparse
import os
import time
import pandas as pd

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

from utils import TestLoader, index_mapping, Verification_Loader
from model import ResNet

torch.set_num_threads(8)


def test_class(data_loader, model, device):
    model.eval()
    
    name_list = []
    pred_list = []

    for name, figs in data_loader:
        
        figs = figs.to(device)

        with torch.no_grad():
            pred = model(figs)

        _, pred_labels = torch.max(F.softmax(pred, dim=1),1)
        pred_labels.view(-1)
        
        name_list.append(name)
        pred_list.append(pred_labels.cpu().numpy())
        
    return name_list, pred_list

def verification(data_loader, model, device, mode):
    model.eval()
    model.change_task('Verification')

    name_list1 = []
    name_list2 = []
    score_list = []
    truth_list = []
    if mode == 'val':
        
        for fig1, fig2, name1, name2, truth in data_loader:
            
            fig1 = fig1.to(device)
            fig2 = fig2.to(device)

            with torch.no_grad():
                feat1 = model(fig1).cpu().numpy()
                feat2 = model(fig2).cpu().numpy()

            value = cosine_similarity(feat1,feat2).diagonal()

            # import pdb; pdb.set_trace()

            name_list1.append(name1)
            name_list2.append(name2)
            score_list.append(value)
            truth_list.append(truth)
    else:
        count = 0
        for fig1, fig2, name1, name2 in data_loader:
            
            fig1 = fig1.to(device)
            fig2 = fig2.to(device)

            with torch.no_grad():
                feat1 = model(fig1).cpu().numpy()
                feat2 = model(fig2).cpu().numpy()

            value = cosine_similarity(feat1,feat2).diagonal()

            # import pdb; pdb.set_trace()

            name_list1.append(name1)
            name_list2.append(name2)
            score_list.append(value)
            count += 1
            print(count)
        
    return name_list1, name_list2, score_list, truth_list


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int, default=256, help='Batch size')
    parser.add_argument('--save_path',type=str,default='/home/yihe/Data/hw2/model/')
    parser.add_argument('--class_folder',type=str,default='/home/yihe/Data/hw2/11-785hw2p2-s20/test_classification/medium')
    parser.add_argument('--mode',type=str,default='val')

    args = parser.parse_args()

    if args.mode == 'test':
        ver_folder = '/home/yihe/Data/hw2/11-785hw2p2-s20/test_verification/'
        ver_file = '/home/yihe/Data/hw2/11-785hw2p2-s20/test_trials_verification_student.txt'
    else:
        ver_folder = '/home/yihe/Data/hw2/11-785hw2p2-s20/validation_verification/'
        ver_file = '/home/yihe/Data/hw2/11-785hw2p2-s20/validation_trials_verification.txt'
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_dataset = TestLoader(
        path=args.class_folder,
        transform=preprocess
    )

    class_loader = DataLoader(
        class_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    ver_dataset = Verification_Loader(
        data_path = ver_folder,
        order_file=ver_file,
        transform=preprocess, 
        mode=args.mode
    )

    ver_loader = DataLoader(
        ver_dataset,
        batch_size = 8,
        shuffle=False
    )

    print(f"Verification Dataset Length: {len(ver_dataset)}")

    #Hyperparameters
    num_channel = 3
    num_class = 2300
    batch_size = args.batch_size
    hidden_sizes = [3,4,6,3]

    index_dict = index_mapping()

    model = ResNet(num_channel,num_class,hidden_sizes)
    model.load_state_dict(torch.load('65_model.pth.tar'))
    model.to(device)

    names, preds = test_class(class_loader, model, device)
    names = np.hstack(names)
    preds = np.hstack(preds)
    true_preds = []
    for pred in preds:
        true_preds.append(index_dict[pred])

    class_results = pd.DataFrame({'Id':names, 'Category': true_preds})
    class_results.to_csv('class.csv', index=False)

    # names1, names2, score_list, truth_list = verification(ver_loader, model, device, mode=args.mode)
    # import pdb; pdb.set_trace()
    # names1 = np.hstack(names1)
    # names2 = np.hstack(names2)
    # scores = np.hstack(score_list)
    # if len(truth_list) > 0:
    #     truth = np.hstack(truth_list)

    # names = np.core.defchararray.add(names1, ' ')
    # names = np.core.defchararray.add(names, names2)
    # ver_results = pd.DataFrame({'trial': names, 'score':scores})
    # ver_results.to_csv('ver.csv',index=False)

    # if args.mode == 'val':
    #     ver_truth = pd.DataFrame({'trial': names, 'score':truth})
    #     ver_truth.to_csv('truth.csv',index=False)

    # import pdb; pdb.set_trace()




    

    

