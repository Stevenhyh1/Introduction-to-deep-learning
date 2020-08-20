import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
letter_list = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ', '<sos>', '<eos>']


def create_list(train_label_path):
    train_label = np.load(train_label_path, allow_pickle=True)

    char_set = set()
    for article in train_label:
        for word in article:
            word = word.decode('UTF-8')
            for char in word:
                char_set.add(char)
    char_list = list(char_set)
    char_list.sort()
    char_list.insert(0, '<pad>')
    char_list.append(' ')
    char_list.append('<sos>')
    char_list.append('<eos>')

    return char_list


def create_dict(letter_list):
    letter2index = dict()
    index2letter = dict()
    for i in range(len(letter_list)):
        letter2index[letter_list[i]] = i
        index2letter[i] = letter_list[i]
    return letter2index, index2letter


def transcript_encoding(transcripts, letter2index):
    labels_list = []
    for transcript in transcripts:
        labels_list.append(transcript2index(transcript, letter2index))

    return labels_list


def transcript2index(transcript, letter2index):
    index_list = []
    for i, word in enumerate(transcript):
        word = word.decode('UTF-8')
        for letter in word:
            index_list.append(letter2index[letter])
        if i is not len(transcript) - 1:
            index_list.append(letter2index[' '])
    index_list.insert(0, letter2index['<sos>'])
    index_list.append(letter2index['<eos>'])
    return index_list


def collate_wrapper(batch_data):
    inputs, targets = zip(*batch_data)

    batch_size = len(inputs)
    feature_dim = inputs[0].shape[-1]

    inputs = list(inputs)
    input_lens = [len(_input) for _input in inputs]
    longest_input = max(input_lens)
    padded_input = torch.zeros(batch_size, longest_input, feature_dim)
    for i, input_len in enumerate(input_lens):
        cur_input = inputs[i]
        padded_input[i, 0:input_len] = cur_input
    padded_input = padded_input.permute(1, 0, 2)  # (T, B, D)

    targets = list(targets)
    target_lens = [len(label)-1 for label in targets]
    longest_target = max(target_lens)
    padded_decoder = torch.zeros((batch_size, longest_target), dtype=torch.int64)
    padded_target = torch.zeros((batch_size, longest_target), dtype=torch.int64)  # (B, T)
    for i, target_len in enumerate(target_lens):
        cur_target = targets[i]
        padded_decoder[i, 0:target_len] = cur_target[:-1]
        padded_target[i, 0:target_len] = cur_target[1:]

    return padded_input, padded_target, padded_decoder, input_lens, target_lens


def collate_wrapper_test(batch_data):
    inputs = batch_data

    batch_size = len(inputs)
    feature_dim = inputs[0].shape[-1]

    inputs = list(inputs)
    input_lens = [len(_input) for _input in inputs]
    longest_input = max(input_lens)
    padded_input = torch.zeros(batch_size, longest_input, feature_dim)
    for i, input_len in enumerate(input_lens):
        cur_input = inputs[i]
        padded_input[i, 0:input_len] = cur_input
    padded_input = padded_input.permute(1, 0, 2)

    return padded_input, input_lens


def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    data_array = []
    eps = 1e-8
    for i in range(len(data)):
        cur_data = data[i]
        mean = np.mean(cur_data, axis=1).reshape(-1, 1)
        var = np.var(cur_data, axis=1).reshape(-1, 1)
        data_array.append((cur_data-mean)/np.sqrt(var+eps))
    return data_array


def load_label(file_path):
    label = np.load(file_path, allow_pickle=True, encoding='bytes')
    return list(label)


class SpeechDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = list(data)
        self.label = label
        if label is not None:
            self.label = list(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label is None:
            return torch.tensor(self.data[idx])
        return torch.tensor(self.data[idx]), torch.tensor(self.label[idx])