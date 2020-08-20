import argparse
import os
import pkbar
import time
import numpy as np
from Levenshtein import distance

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Seq2Seq
from dataloader import SpeechDataset, load_label, load_data
from dataloader import collate_wrapper, create_dict, transcript_encoding

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(6)


def train(model, train_loader, criterion, optimizer, epoch, train_batch_num, writer):
    train_start = time.time()
    model.train()

    epoch_perplexity = 0
    epoch_loss = 0

    kbar = pkbar.Kbar(train_batch_num)

    for batch, (padded_input, padded_target, padded_decoder, input_lens, target_lens) in enumerate(train_loader):
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            batch_size = len(input_lens)
            vocab_size = model.vocab_size
            max_len = max(target_lens)

            padded_input = padded_input.to(DEVICE)
            padded_target = padded_target.type(torch.LongTensor).to(DEVICE)
            padded_decoder = padded_decoder.type(torch.LongTensor).to(DEVICE)

            predictions = model(padded_input, input_lens, epoch, padded_decoder)

            mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(target_lens).unsqueeze(1)
            mask = mask.type(torch.float64)
            mask.requires_grad = True
            mask = mask.reshape(batch_size * max_len).to(DEVICE)

            predictions = predictions.reshape(batch_size * max_len, vocab_size).contiguous()
            padded_target = padded_target.reshape(batch_size * max_len).contiguous()

            loss = criterion(predictions, padded_target)
            masked_loss = torch.sum(loss * mask)
            batch_loss = masked_loss / torch.sum(mask).item()
            batch_loss.backward()
            epoch_loss += batch_loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            perplexity = np.exp(batch_loss.item())
            epoch_perplexity += perplexity
            kbar.update(batch, values=[("loss", batch_loss)])

    kbar.add(1)
    writer.add_scalar('Loss/Train', epoch_loss / train_batch_num, epoch)
    writer.add_scalar('Perplexity/Train', epoch_perplexity / train_batch_num, epoch)

    return epoch_loss/train_batch_num


def val(model, val_loader, criterion, epoch, val_batch_num, index2letter, writer):
    model.eval()

    epoch_distance = 0
    epoch_perplexity = 0
    epoch_loss = 0

    kbar = pkbar.Kbar(val_batch_num)

    for batch, (padded_input, padded_target, padded_decoder, input_lens, target_lens) in enumerate(val_loader):

        with torch.no_grad():

            batch_size = len(input_lens)
            vocab_size = model.vocab_size
            max_len = max(target_lens)

            padded_input = padded_input.to(DEVICE)
            padded_target = padded_target.to(DEVICE)
            padded_decoder = padded_decoder.to(DEVICE)

            predictions = model(padded_input, input_lens, epoch, padded_decoder)
            inferences = torch.argmax(predictions, dim=2)
            targets = padded_target

            mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(target_lens).unsqueeze(1)
            mask = mask.type(torch.float64)
            mask = mask.reshape(batch_size * max_len).to(DEVICE)

            predictions = predictions.reshape(batch_size * max_len, vocab_size)
            padded_target = padded_target.reshape(batch_size * max_len)

            loss = criterion(predictions, padded_target)
            masked_loss = torch.sum(loss * mask)
            batch_loss = masked_loss / torch.sum(mask).item()
            epoch_loss += batch_loss.item()
            perplexity = np.exp(batch_loss.item())
            epoch_perplexity += perplexity

            cur_dis = 0
            for i, article in enumerate(inferences):
                inference = ''
                for k in article:
                    inference += index2letter[k.item()]
                    if index2letter[k.item()] == '<eos>':
                        break
                target = ''.join(index2letter[k.item()] for k in targets[i])
                if i == len(inferences) - 1 and batch == val_batch_num - 1:
                    print('\nInput text:\n', target[:150])
                    print('Pred text:\n', inference[:150])
                cur_dis += distance(inference, target)
            batch_dis = cur_dis / batch_size

            epoch_distance += batch_dis
            kbar.update(batch, values=[("loss", batch_loss), ("Dis", batch_dis)])

    kbar.add(1)
    writer.add_scalar('Loss/Val', epoch_loss / val_batch_num, epoch)
    writer.add_scalar('Perplexity/Val', epoch_perplexity / val_batch_num, epoch)
    writer.add_scalar('Distance/val', epoch_distance / val_batch_num, epoch)

    return epoch_loss/val_batch_num


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dim', type=int, default=40, help='Input dimenstion of model')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of LSTM')
    parser.add_argument('--encoder_layers', type=int, default=3, help='LSTM layers')
    parser.add_argument('--value_size', type=int, default=256, help='Value size of attention')
    parser.add_argument('--key_size', type=int, default=256, help='Key size of attention')

    parser.add_argument('--num_epoch', type=int, default=200, help='Number of Epoch')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weightdecay')

    parser.add_argument('--data_root_dir', type=str, default='/home/yihe/Data/hw4p2/', help='Root directory')
    parser.add_argument('--train_data_path', type=str, default='train_new.npy')
    parser.add_argument('--train_label_path', type=str, default='train_transcripts.npy')
    parser.add_argument('--dev_data_path', type=str, default='dev_new.npy')
    parser.add_argument('--dev_label_path', type=str, default='dev_transcripts.npy')
    parser.add_argument('--log_dir', type=str, default='runs/256_3_256')
    parser.add_argument('--model_dir', type=str, default='256_3_256')
    args = parser.parse_args()

    train_data_path = os.path.join(args.data_root_dir, args.train_data_path)
    train_label_path = os.path.join(args.data_root_dir, args.train_label_path)
    val_data_path = os.path.join(args.data_root_dir, args.dev_data_path)
    val_label_path = os.path.join(args.data_root_dir, args.dev_label_path)

    letter_list = ['<pad>', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                   'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', ' ', '<sos>', '<eos>']
    letter2index, index2letter = create_dict(letter_list)

    print('Load Training Data')
    train_data = load_data(train_data_path)
    train_transcript = load_label(train_label_path)
    train_label = transcript_encoding(train_transcript, letter2index)
    train_dataset = SpeechDataset(train_data, train_label)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_wrapper
    )
    train_batch_num = int(len(train_data) / args.batch_size) + 1

    print('Load Validation Data')
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
    val_batch_num = int(len(val_data) / args.batch_size) + 1
    print('Done')

    model = Seq2Seq(input_dim=args.input_dim, vocab_size=len(letter_list), hidden_dim=args.hidden_dim,
                    value_size=args.value_size, key_size=args.key_size, pyramidlayers=args.encoder_layers)
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.num_epoch)
    criterion = nn.CrossEntropyLoss(reduction='none').to(DEVICE)

    writer = SummaryWriter(args.log_dir)
    best_perp = 1000
    for epoch in range(args.num_epoch):
        epoch += 1
        print(f'Epoch {epoch} starts:')
        train_perp = train(model, train_loader, criterion, optimizer, epoch, train_batch_num, writer)
        scheduler.step()
        val_perp = val(model, val_loader, criterion, epoch, val_batch_num, index2letter, writer)
        if val_perp < best_perp:
            torch.save(model.state_dict(), f"./models/{args.model_dir}/{epoch}_model.pth.tar")
            best_perp = val_perp
