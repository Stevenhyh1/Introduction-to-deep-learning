import argparse
import os
import pkbar
import numpy as np
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch.utils.data import DataLoader
from model import Seq2Seq
from dataloader import SpeechDataset, load_data
from dataloader import collate_wrapper_test, create_dict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(6)

parser = argparse.ArgumentParser()

parser.add_argument('--input_dim', type=int, default=40, help='Input dimenstion of model')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of LSTM')
parser.add_argument('--encoder_layers', type=int, default=4, help='LSTM layers')
parser.add_argument('--value_size', type=int, default=128, help='Value size of attention')
parser.add_argument('--key_size', type=int, default=128, help='Key size of attention')

parser.add_argument('--data_root_dir', type=str, default='/home/yihe/Data/hw4p2/', help='Root directory')
parser.add_argument('--train_data_path', type=str, default='train_new.npy')
parser.add_argument('--train_label_path', type=str, default='train_transcripts.npy')
parser.add_argument('--dev_data_path', type=str, default='dev_new.npy')
parser.add_argument('--dev_label_path', type=str, default='dev_transcripts.npy')
parser.add_argument('--inf_data_path', type=str, default='test_new.npy')

parser.add_argument('--log_dir', type=str, default='./Tensorboard/')
args = parser.parse_args()


def test(model, val_loader, index2letter, pbar):
    model.eval()
    prediction_list = []

    for batch, (padded_input, input_lens) in enumerate(val_loader):

        with torch.no_grad():

            padded_input = padded_input.to(DEVICE)

            predictions = model(padded_input, input_lens, epoch=0, text_input=None, istrain=False)
            inferences = torch.argmax(predictions, dim=2)
            inferences = inferences.squeeze(0)

            inference = ''
            for i, char in enumerate(inferences):
                if index2letter[char.item()] == '<eos>':
                    break
                inference += index2letter[char.item()]
            print(inference)
            prediction_list.append(inference)

    return prediction_list


if __name__ == '__main__':
    letter_list = ['<pad>', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                   'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', ' ', '<sos>', '<eos>']
    letter2index, index2letter = create_dict(letter_list)

    inf_data_path = os.path.join(args.data_root_dir, args.inf_data_path)
    print('Load Testing Data')
    inf_data = load_data(inf_data_path)
    inf_dataset = SpeechDataset(inf_data)
    inf_loader = DataLoader(
        inf_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_wrapper_test
    )
    pbar = pkbar.Pbar(name='Inference', target=len(inf_data))

    model = Seq2Seq(input_dim=args.input_dim, vocab_size=len(letter_list), hidden_dim=args.hidden_dim,
                    value_size=args.value_size, key_size=args.key_size, pyramidlayers=args.encoder_layers)
    model.to(DEVICE)
    model.load_state_dict(torch.load('./models/76_model.pth.tar'))
    prediction_list = test(model, inf_loader, index2letter, pbar)

    index = np.linspace(0, len(prediction_list) - 1, len(prediction_list)).astype(np.int32)
    test_result = pd.DataFrame({'Id': index, 'Predicted': prediction_list})
    print('Saving predictions...')
    test_result.to_csv('Test.csv', index=False)
    print('Done')
