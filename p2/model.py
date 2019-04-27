import random
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F
from config import Config
from character_list import CHARACTER_LIST


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
        lengths = [l // 2 for l in lengths]
        packed_input = rnn.pack_padded_sequence(padded_input, lengths, batch_first=True)
        output_packed, hidden = self.blstm(packed_input)
        output_padded, _ = rnn.pad_packed_sequence(output_packed, batch_first=True)
        return output_padded, hidden, lengths


class Listener(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Listener, self).__init__()
        self.pblstm1 = pBLSTM(input_dim, hidden_dim)
        self.pblstm2 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.pblstm3 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x, lengths):
        # x shape: (batch_size, length, dim)
        x, hidden, lengths = self.pblstm1(x, lengths)
        x, hidden, lengths = self.pblstm2(x, lengths)
        x, hidden, lengths = self.pblstm3(x, lengths)
        x = self.dropout2(x)
        return x, hidden


class Speller(nn.Module):
    def __init__(self, hidden_size, embed_size, context_size, output_size, num_layer=2, max_steps=250):
        super(Speller, self).__init__()
        self.hidden_size = hidden_size
        self.max_steps = max_steps

        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.ModuleList()
        for i in range(num_layer):
            if i == 0:
                self.rnn.append(nn.LSTMCell(hidden_size + context_size, hidden_size))
            else:
                self.rnn.append(nn.LSTMCell(hidden_size, hidden_size))
        self.character_distribution = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, listener_output, teacher_forcing_ratio, ground_truth=None):
        if ground_truth is None:
            teacher_forcing_ratio = 0
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        # input shape: (batch_size, length, listener_hidden_dim * 2)
        batch_size = listener_output.shape[0]
        output_char = torch.zeros(batch_size).fill_(CHARACTER_LIST.index(Config.EOS)).to(Config.DEVICE)
        states = [None, None]
        outputs = []
        for i in range(self.max_steps) if ground_truth is None else range(ground_truth.shape[1]):
            output, states = self.forward_step(listener_output, output_char, states)
            outputs.append(output)
            if use_teacher_forcing:
                output_char = ground_truth[:, i]
            else:
                _, output_char = torch.max(output, dim=1)
        return torch.stack(outputs, dim=1)

    def forward_step(self, listener_output, last_char, states):
        embed = self.embedding(last_char.long())
        # rnn_input = torch.cat((embed, context), dim=1)
        rnn_input = embed
        new_states = []
        for i, cell in enumerate(self.rnn):
            state = cell(rnn_input, states[i])
            new_states.append(state)
        rnn_output = new_states[-1][0]  # hidden state of the last RNN layer
        output = self.softmax(self.character_distribution(rnn_output))
        return output, new_states
