
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from Levenshtein import distance as levenshtein_distance
from ctcdecode import CTCBeamDecoder

from utils.dataloader import SpeechDataset, load_label, load_data
from utils.dataprocess import collate_wrapper
from models.Baseline import BiLSTM
from config.phoneme_list import PHONEME_LIST, PHONEME_MAP
phonemes = PHONEME_MAP
phonemes.insert(0,' ')

torch.set_num_threads(4)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weightdecay')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hideen dimension of LSTM')
parser.add_argument('--hidden_layer', type=int, default=3, help='LSTM layers')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of Epoch')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
# parser.add_argument('--train_data_path', type=str, default='/home/yihe/Data/hw3p2/wsj0_train')
# parser.add_argument('--train_label_path', type=str, default='/home/yihe/Data/hw3p2/wsj0_train_merged_labels.npy')
parser.add_argument('--train_data_path', type=str, default='/home/yihe/Data/hw3p2/wsj0_dev.npy')
parser.add_argument('--train_label_path', type=str, default='/home/yihe/Data/hw3p2/wsj0_dev_merged_labels.npy')
parser.add_argument('--dev_data_path', type=str, default='/home/yihe/Data/hw3p2/wsj0_dev.npy')
parser.add_argument('--dev_label_path', type=str, default='/home/yihe/Data/hw3p2/wsj0_dev_merged_labels.npy')
parser.add_argument('--k', type=int, default=12)
args = parser.parse_args()

def Train(data, model, criterion, optimizer):
    epoch_loss = 0
    count = 0
    model.train()

    for padded_input, padded_target, input_lens, target_lens in data:
        
        # import pdb; pdb.set_trace()
        optimizer.zero_grad()
        batch_size = len(input_lens)
        inputs = padded_input.to(device)
        targets = padded_target.to(device)

        pred, pred_lens = model(inputs.float(),input_lens)
        pred_lens = pred_lens.tolist()

        loss = criterion(pred, targets, pred_lens, target_lens)
        epoch_loss += loss.detach()/batch_size
        loss.backward()
        optimizer.step()
        count += 1
    
    return epoch_loss/count

def Val(data, model, criterion, decoder):
    epoch_loss = 0
    count = 0
    epoch_dis = 0
    model.train()

    for padded_input, padded_target, input_lens, target_lens in data:
        
        batch_size = len(input_lens)
        inputs = padded_input.to(device)
        targets = padded_target.to(device)
        
        with torch.no_grad():
            pred, pred_lens = model(inputs.float(),input_lens)
        pred_lens_loss = pred_lens.tolist()
        loss = criterion(pred, targets, pred_lens_loss, target_lens)
        epoch_loss += loss.detach()/batch_size
        count += 1

        probs = pred.transpose(0,1)
        # import pdb; pdb.set_trace()
        out, _, _, out_lens = decoder.decode(probs, pred_lens)
        cur_dist = 0
        for i in range(batch_size):
            best_seq = out[i,0, :out_lens[i,0]]
            best_phenome = ''
            for j in best_seq:
                best_phenome += phonemes[j] 
            # import pdb; pdb.set_trace()
            target_seq = targets[i].int()
            target_phenome = ''
            for j in target_seq:
                target_phenome += phonemes[j] 
            cur_dist += levenshtein_distance(best_phenome, target_phenome)
        # import pdb; pdb.set_trace()
        cur_dist /= batch_size
        epoch_dis += cur_dist

    return epoch_loss/count, epoch_dis/count


if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    torch.manual_seed(11785)

    train_data_path = args.train_data_path
    train_label_path = args.train_label_path
    val_data_path = args.dev_data_path
    val_label_path = args.dev_label_path
    input_dim = 40
    output_dim = 47

    print('Loading training data... ')
    train_data = load_data(train_data_path)
    train_label = load_label(train_label_path)
    train_dataset = SpeechDataset(train_data, train_label)
    train_loader = DataLoader(
        train_dataset, 
        batch_size = args.batch_size,
        shuffle = True,
        collate_fn=collate_wrapper
    )

    print('Loading validation data... ')
    val_data = load_data(val_data_path)
    val_label = load_label(val_label_path)
    val_dataset = SpeechDataset(val_data, val_label)
    val_loader = DataLoader(
        val_dataset, 
        batch_size = args.batch_size,
        shuffle = False,
        collate_fn=collate_wrapper
    )

    print('Data loaded!')

    model = BiLSTM(input_dim, args.hidden_dim, output_dim, args.hidden_layer, args.dropout)
    model.to(device)
    criterion = nn.CTCLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    decoder = CTCBeamDecoder(phonemes, beam_width=10, num_processes=4, log_probs_input=True)
    # import pdb; pdb.set_trace()

    global_dis = 10000
    for epoch in range(args.num_epoch):
        epoch = epoch+1
        print(f'Epoch {epoch} starts:')
        train_start = time.time()
        train_loss = Train(
            train_loader,
            model,
            criterion,
            optimizer
        )
        train_end = time.time()
        scheduler.step()
        print(f"Epoch {epoch} completed in: {train_end-train_start}s \t Loss: {train_loss}")

        val_start = time.time()
        val_loss, val_dis = Val(
            val_loader,
            model,
            criterion,
            decoder
        )
        val_end = time.time()
        if val_dis < global_dis:
            torch.save(model.state_dict(), f".model/{epoch}_model.pth.tar")
            global_dis = val_dis
        print(f"Validation Loss: {val_loss} \t Validation Dis: {val_dis}")