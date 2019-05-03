import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F
from config import Config
from character_list import CHARACTER_LIST


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


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
        self.lockdropout = LockedDropout()
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x, lengths):
        # x shape: (batch_size, length, dim)
        x, hidden, lengths = self.pblstm1(x, lengths)
        x = self.lockdropout(x, 0.1)
        x, hidden, lengths = self.pblstm2(x, lengths)
        x = self.lockdropout(x, 0.1)
        x, hidden, lengths = self.pblstm3(x, lengths)
        x = self.lockdropout(x, 0.2)
        # x = self.dropout2(x)
        return x, hidden, lengths


class Speller(nn.Module):
    def __init__(self, hidden_size, embed_size, context_size, output_size, attention, num_layer=2, max_steps=250):
        super(Speller, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.context_size = context_size
        self.output_size = output_size
        self.max_steps = max_steps
        self.attention = attention
        self.num_layer = num_layer

        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.ModuleList()
        self.inith = nn.ParameterList()
        self.initc = nn.ParameterList()
        for i in range(num_layer):
            if i == 0:
                self.rnn.append(nn.LSTMCell(embed_size + context_size, hidden_size))
            else:
                self.rnn.append(nn.LSTMCell(hidden_size, hidden_size))
            self.inith.append(nn.Parameter(torch.rand(1, hidden_size)))
            self.initc.append(nn.Parameter(torch.rand(1, hidden_size)))
        self.unembed = nn.Linear(hidden_size, output_size)
        self.unembed.weight = self.embedding.weight
        self.character_distribution = nn.Sequential(
            nn.Linear(hidden_size + context_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
            self.unembed)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, listener_output, teacher_forcing_ratio, lengths, ground_truth=None):
        if ground_truth is None:
            teacher_forcing_ratio = 0
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        # listener_output shape: (batch_size, length, listener_hidden_dim * 2)
        batch_size = listener_output.shape[0]
        output_char = torch.zeros(batch_size).fill_(CHARACTER_LIST.index(Config.EOS)).to(Config.DEVICE)

        rnn_hidden = [h.repeat(batch_size, 1) for h in self.inith]
        rnn_cell = [c.repeat(batch_size, 1) for c in self.initc]
        states = [rnn_hidden, rnn_cell]
        outputs = []
        attentions = []

        for i in range(self.max_steps) if ground_truth is None else range(ground_truth.shape[1]):
            output, masked_attention, states = self.forward_step(listener_output, output_char, states, lengths)
            outputs.append(output)
            attentions.append(masked_attention)
            if use_teacher_forcing:
                output_char = ground_truth[:, i]
            else:
                if ground_truth is not None:
                    _, output_char = torch.max(self.softmax(output + np.random.gumbel()), dim=1)
                else:
                    _, output_char = torch.max(self.softmax(output), dim=1)
        return torch.stack(outputs, dim=1), attentions

    def forward_step(self, listener_output, last_char, states, lengths):
        embed = self.embedding(last_char.long())
        # embed shape: (batch_size, SPELLER_EMBED_SIZE)
        old_hidden, old_cell = states[0], states[1]
        context, masked_attention = self.attention(listener_output, old_hidden[-1], lengths)
        # context shape: (batch_size, CONTEXT_SIZE)
        rnn_input = torch.cat((embed, context), dim=1)
        # rnn_input: (batch_size, SPELLER_EMBED_SIZE + CONTEXT_SIZE)
        new_hidden, new_cell = [None] * self.num_layer, [None] * self.num_layer
        for i, cell in enumerate(self.rnn):
            new_hidden[i], new_cell[i] = cell(rnn_input, (old_hidden[i], old_cell[i]))
            rnn_input = new_hidden[i]
        rnn_output = new_hidden[-1]  # hidden state of the last RNN layer
        concat_output = torch.cat((rnn_output, context), dim=1)
        output = self.character_distribution(concat_output)
        return output, masked_attention, [new_hidden, new_cell]


class Attention(nn.Module):
    def __init__(self, key_query_val_dim, context_dim, listener_dim, speller_dim):
        super(Attention, self).__init__()
        self.query_fc = nn.Linear(speller_dim, key_query_val_dim)  # mlp for query
        self.key_fc = nn.Linear(listener_dim, key_query_val_dim)  # mlp for key
        self.value_fc = nn.Linear(listener_dim, context_dim)  # mlp for value

        self.softmax = nn.Softmax()
        self.activate = torch.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, listener_output, decoder_state, lengths):
        # listener_output shape: (batch_size, length, LISTENER_HIDDEN_SIZE * 2)
        # decoder_state shape: (batch_size, SPELLER_HIDDEN_SIZE)
        # lengths shape: (batch_size, max_seq_len)
        query = self.activate(self.query_fc(decoder_state).unsqueeze(1))  # (batch_size, 1, KEY_QUERY_VAL_DIM)
        key = self.activate(self.key_fc(listener_output))  # (batch_size, length, KEY_QUERY_VAL_DIM)
        value = self.activate(self.value_fc(listener_output))  # (batch_size, length, CONTEXT_SIZE)
        energy = torch.bmm(query, key.transpose(1, 2)).squeeze(1)
        # energy shape: (batch_size, length)
        attention = self.softmax(energy)
        # TODO: mask shape and value check
        mask = attention.data.new(attention.size(0), attention.size(1)).zero_()
        for i, len in enumerate(lengths):
            mask[i, :len] = 1
        masked_attention = F.normalize(mask * attention, p=1)
        # TODO: check for masked_attention
        context = torch.bmm(masked_attention.unsqueeze(1), value).squeeze(1)
        return context, masked_attention


class LAS(nn.Module):
    def __init__(self):
        super(LAS, self).__init__()
        self.listener = Listener(input_dim=Config.INPUT_DIM, hidden_dim=Config.LISTENER_HIDDEN_SIZE)
        self.attention = Attention(key_query_val_dim=Config.KEY_QUERY_VAL_SIZE,
                                   context_dim=Config.CONTEXT_SIZE,
                                   listener_dim=Config.LISTENER_HIDDEN_SIZE * 2,
                                   speller_dim=Config.SPELLER_HIDDEN_SIZE)
        self.speller = Speller(hidden_size=Config.SPELLER_HIDDEN_SIZE,
                               embed_size=Config.SPELLER_EMBED_SIZE,
                               context_size=Config.CONTEXT_SIZE,
                               output_size=Config.NUM_CLASS,
                               attention=self.attention)

    def forward(self, inputs, labels, lengths, teacher_forcing_ratio):
        encoder_outputs, hidden, lengths = self.listener(inputs, lengths)
        decoder_outputs, attentions = self.speller(encoder_outputs, teacher_forcing_ratio, lengths, labels)
        return decoder_outputs, attentions
