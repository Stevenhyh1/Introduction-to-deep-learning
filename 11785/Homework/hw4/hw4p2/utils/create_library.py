import numpy as np


def create_dict(train_label_path):
    # train_label_path = '/home/yihe/Data/hw4p2/train_transcripts.npy'
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
    char_list.append('<eof>')

    return char_list

