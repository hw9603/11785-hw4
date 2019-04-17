import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from config import Config
from character_list import CHARACTER_LIST


class SpeechDataset(Dataset):
    def __init__(self, x, y=None, is_test=False):
        self.is_test = is_test
        self.x = [torch.tensor(x_) for x_ in x]
        if not is_test:
            self.total_words = sum(len(y_) for y_ in y)
            self.y = [torch.tensor(y_) for y_ in y]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        if not self.is_test:
            labels = []
            for word in self.y[item]:
                for c in word:
                    labels.append(CHARACTER_LIST.index(chr(c)))
                labels.append(CHARACTER_LIST.index(" "))
            labels[-1] = CHARACTER_LIST.index(Config.EOS)
            return self.x[item].to(Config.DEVICE), labels.to(Config.DEVICE)
        else:
            return self.x[item].to(Config.DEVICE), torch.tensor([-1]).to(Config.DEVICE)


def collate_speech(seq_list):
    seq_list = sorted(seq_list, key=lambda seq: seq[0].shape[0], reverse=True)
    x = []
    y = []
    x_lens = []
    y_lens = []
    for i, (x_, y_) in enumerate(seq_list):
        x.append(x_)
        y.append(y_)
        x_lens.append(x_.shape[0])
        y_lens.append(y_.shape[0])
    return x, y, x_lens, y_lens


def get_loaders():
    data_path = "data/"
    train_x_file = data_path + "train.npy"
    train_y_file = data_path + "train_transcripts.npy"
    dev_x_file = data_path + "dev.npy"
    dev_y_file = data_path + "dev_transcripts.npy"
    test_x_file = data_path + "test.npy"

    train_x = np.load(train_x_file, encoding='bytes')
    train_y = np.load(train_y_file, encoding='bytes')
    dev_x = np.load(dev_x_file, encoding='bytes')
    dev_y = np.load(dev_y_file, encoding='bytes')
    test_x = np.load(test_x_file, encoding='bytes')

    train_dataset = SpeechDataset(train_x, train_y, is_test=False)
    dev_dataset = SpeechDataset(dev_x, dev_y, is_test=False)
    test_dataset = SpeechDataset(test_x, y=None, is_test=True)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=Config.BATCH_SIZE, collate_fn=collate_speech)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=Config.BATCH_SIZE, collate_fn=collate_speech)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate_speech)

    return train_loader, dev_loader, test_loader
