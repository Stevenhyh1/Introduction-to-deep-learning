import torch
import numpy as np


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
    target_lens = [len(label) for label in targets]
    longest_target = max(target_lens)
    padded_target = torch.zeros(batch_size, longest_target)  # (B, T)
    for i, target_len in enumerate(target_lens):
        cur_target = targets[i]
        padded_target[i, 0:target_len] = cur_target

    return padded_input, padded_target, input_lens, target_lens


def collate_wrapper_test(batch_data):
    inputs = batch_data
    # import pdb; pdb.set_trace()

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
