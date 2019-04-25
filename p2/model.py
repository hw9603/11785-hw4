import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F


class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_dim * 2, hidden_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        batch_size = len(x)
        packed_input = rnn.pack_sequence(x)
        seq_length = packed_input[0].shape[0]
        feature_dim = packed_input[0].shape[1]
        # reduce the timestep
        packed_input = packed_input.contiguous().view(batch_size, int(seq_length // 2), feature_dim * 2)
        output_packed, hidden = self.blstm(packed_input)  # TODO: what is the shape of hidden?
        output_padded, _ = rnn.pad_packed_sequence(output_packed)
        return output_packed, hidden


class Listener(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Listener, self).__init__()
        self.pblstm1 = pBLSTM(input_dim, hidden_dim)
        self.pblstm2 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.pblstm3 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):
        x, hidden = self.pblstm1(x)
        x, hidden = self.pblstm2(x)
        x, hidden = self.pblstm3(x)
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
