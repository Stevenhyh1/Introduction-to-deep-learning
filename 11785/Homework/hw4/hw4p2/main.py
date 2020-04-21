import argparse
import os
import pkbar

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.LAS import Seq2Seq
from train_test import train, val
from utils.dataloader import SpeechDataset, load_label, load_data
from utils.dataprocess import collate_wrapper, create_list, create_dict, transcript_encoding

torch.set_num_threads(6)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weightdecay')
parser.add_argument('--input_dim', type=int, default=40, help='Input dimenstion of model')
parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of LSTM')
parser.add_argument('--hidden_layer', type=int, default=3, help='LSTM layers')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of Epoch')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--data_root_dir', type=str, default='/media/yihe/HDD/Data/hw4p2', help='Root directory')
parser.add_argument('--train_data_path', type=str, default='train_new.npy')
parser.add_argument('--train_label_path', type=str, default='train_transcripts.npy')
parser.add_argument('--dev_data_path', type=str, default='dev_new.npy')
parser.add_argument('--dev_label_path', type=str, default='dev_transcripts.npy')
parser.add_argument('--log_dir', type=str, default='./Tensorboard/')
args = parser.parse_args()

if __name__ == "__main__":

    torch.manual_seed(11785)

    train_data_path = os.path.join(args.data_root_dir, args.train_data_path)
    train_label_path = os.path.join(args.data_root_dir, args.train_label_path)
    val_data_path = os.path.join(args.data_root_dir, args.dev_data_path)
    val_label_path = os.path.join(args.data_root_dir, args.dev_label_path)

    letter_list = create_list(train_label_path)
    letter2index, index2letter = create_dict(letter_list)
    # print(letter_list)
    # print(letter2index)
    # print(index2letter)
    # ['<pad>', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
    #  'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', ' ', '<sos>', '<eos>']

    print('Loading Training Data')
    train_data = load_data(train_data_path)
    train_transcript = load_label(train_label_path)
    train_label = transcript_encoding(train_transcript, letter2index)
    # print(train_data[0])
    # print(train_label[0])
    train_dataset = SpeechDataset(train_data, train_label)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_wrapper
    )

    print('Loading Validation Data')
    val_data = load_data(val_data_path)
    val_transcript = load_label(val_label_path)
    val_label = transcript_encoding(val_transcript, letter2index)
    val_dataset = SpeechDataset(val_data, val_label)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_wrapper
    )
    print('Done')

    model = Seq2Seq(input_dim=args.input_dim, vocab_size=len(letter_list), hidden_dim=args.hidden_dim,
                    value_size=128, key_size=128, pyramidlayers=3, useattention=True)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    writer = SummaryWriter('./Tensorboard')

    for epoch in range(args.num_epoch):
        epoch += 1
        train(model, train_loader, criterion, optimizer, epoch, DEVICE, writer)
        scheduler.step()
        val(model, val_loader, criterion, epoch, DEVICE, writer, index2letter)
