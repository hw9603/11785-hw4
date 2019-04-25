import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F
from config import Config


class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_dim * 2, hidden_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        feature_dim = x.shape[2]
        if seq_length % 2 != 0:
            x = x[:, :-1, :]
            seq_length -= 1
        # reduce the timestep
        padded_input = x.contiguous().view(batch_size, int(seq_length // 2), feature_dim * 2)
        packed_input = rnn.pack_padded_sequence(padded_input, lengths, batch_first=True)
        output_packed, hidden = self.blstm(packed_input)  # TODO: what is the shape of hidden?
        print(output_packed)
        print("+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(hidden)
        output_padded, _ = rnn.pad_packed_sequence(output_packed)
        return output_packed, hidden


class Listener(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Listener, self).__init__()
        self.pblstm1 = pBLSTM(input_dim, hidden_dim)
        self.pblstm2 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.pblstm3 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x, lengths):
        print(lengths)
        x = rnn.pad_sequence(x, batch_first=True)  # (batch_size, length, dim)
        x, hidden = self.pblstm1(x, lengths)
        x, hidden = self.pblstm2(x, lengths)
        x, hidden = self.pblstm3(x, lengths)
        x = self.dropout2(x)
        return x, hidden


class Speller(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Speller, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
