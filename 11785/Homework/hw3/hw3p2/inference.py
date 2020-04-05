
import numpy as np
import pandas as pd
import argparse
import os
import time
import pkbar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from Levenshtein import distance as levenshtein_distance
from ctcdecode import CTCBeamDecoder

from utils.dataloader import SpeechDataset, load_label, load_data
from utils.dataprocess import collate_wrapper, collate_wrapper_test
from models.Baseline import BiLSTM, BiLSTM1
from config.phoneme_list import PHONEME_LIST, PHONEME_MAP
phonemes = PHONEME_MAP
phonemes.insert(0,' ')

torch.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"]="2"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weightdecay')
parser.add_argument('--hidden_dim', type=int, default=512, help='Hideen dimension of LSTM')
parser.add_argument('--hidden_layer', type=int, default=3, help='LSTM layers')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of Epoch')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--dev_data_path', type=str, default='/home/yige/Data/hw3p2/wsj0_dev.npy')
parser.add_argument('--dev_label_path', type=str, default='/home/yige/Data/hw3p2/wsj0_dev_merged_labels.npy')
parser.add_argument('--test_data_path', type=str, default='/home/yige/Data/hw3p2/wsj0_test')
parser.add_argument('--mode', type=str, default='Test', help='Test/Val')

args = parser.parse_args()

def Val(data, model, criterion, decoder, kbar):
    epoch_loss = 0
    count = 0
    epoch_dis = 0
    model.eval()

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

        out, _, _, out_lens = decoder.decode(probs, pred_lens)
        cur_dist = 0
        for i in range(batch_size):
            best_seq = out[i,0, :out_lens[i,0]]
            best_phenome = ''
            for j in best_seq:
                best_phenome += phonemes[j] 

            target_seq = targets[i].int()[:target_lens[i]]
            target_phenome = ''
            for j in target_seq:
                target_phenome += phonemes[j]

            # print("Best Phenome: ", best_phenome)
            # print("Target Pehnome: ", target_phenome)
            # print("Distance: ", levenshtein_distance(best_phenome, target_phenome))
            # import pdb; pdb.set_trace()
            cur_dist += levenshtein_distance(best_phenome, target_phenome)
            kbar.update(count)
        cur_dist /= batch_size
        epoch_dis += cur_dist

    return epoch_loss/count, epoch_dis/count

def Test(data, model, decoder,kbar):
    phenome_list = []
    model.eval()
    count = 0
    
    
    
    for padded_input, input_lens in data:
        
        batch_size = len(input_lens)
        inputs = padded_input.to(device)
        
        with torch.no_grad():
            pred, pred_lens = model(inputs.float(),input_lens)
        probs = pred.transpose(0,1)

        out, _, _, out_lens = decoder.decode(probs, pred_lens)

        for i in range(batch_size):
            best_seq = out[i,0, :out_lens[i,0]]
            best_phenome = ''
            for j in best_seq:
                best_phenome += phonemes[j] 

            # print("Best Phenome: ", best_phenome)
            phenome_list.append(best_phenome)
        count += 1
        kbar.update(count)


    return phenome_list


if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # torch.manual_seed(11785)

    test_data_path = args.test_data_path
    val_data_path = args.dev_data_path
    val_label_path = args.dev_label_path
    input_dim = 40
    output_dim = 47

    if args.mode == 'Test':
        print('Loading test data...')
        test_data = load_data(test_data_path)
        test_dataset = SpeechDataset(test_data)
        test_loader = DataLoader(
            test_dataset, 
            batch_size = args.batch_size,
            shuffle = False,
            collate_fn=collate_wrapper_test
        )
        test_kbar = pkbar.Kbar(int(len(test_data)/args.batch_size)+1)

    else:
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
        val_kbar = pkbar.Kbar(int(len(val_data)/args.batch_size)+1)

    print('Data loaded!')

    model = BiLSTM(input_dim, args.hidden_dim, output_dim, args.hidden_layer, args.dropout)
    model.load_state_dict(torch.load('./model/46_model.pth.tar'))
    model.to(device)
    criterion = nn.CTCLoss()
    decoder = CTCBeamDecoder(phonemes, beam_width=10, num_processes=4, log_probs_input=True)
    

    if args.mode == 'Test':
        phonemes_list = Test(
            test_loader,
            model,
            decoder,
            test_kbar
        )
        index = np.linspace(0, len(phonemes_list)-1,len(phonemes_list)).astype(np.int32)
        test_result = pd.DataFrame({'Id': index, 'Predicted': phonemes_list})
        print('Saving predictions...')
        test_result.to_csv('Test.csv',index=False)
        print('Done')
    else:
        val_start = time.time()
        val_loss, val_dis = Val(
            val_loader,
            model,
            criterion,
            decoder,
            val_kbar
        )
        val_end = time.time()
        print(f"Validation Loss: {val_loss} \t Validation Dis: {val_dis}")

    # import pdb; pdb.set_trace()