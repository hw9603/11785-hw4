import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable


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
        output_packed, _ = self.blstm(packed_input)
        output_padded, _ = rnn.pad_packed_sequence(output_packed)
        return output_packed


class Listener(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Listener, self).__init__()
        self.pblstm1 = pBLSTM(input_dim, hidden_dim)
        self.pblstm2 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.pblstm3 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x, seq_sizes):
        x = self.pblstm1(x)
        x = self.pblstm2(x)
        x = self.pblstm3(x)
        x = self.dropout2(x)

        out_seq_sizes = [size // 8 for size in seq_sizes]
        return x, out_seq_sizes


class Speller(nn.Module):
    def __init__(self, hidden_dim, context_dim, rnn_layer, attention, num_class=33):
        super(Speller, self).__init__()
        self.rnn = []
        for i in range(rnn_layer):
            if i == 0:
                self.rnn.append(nn.LSTM(hidden_dim + context_dim, hidden_dim))
            else:
                self.rnn.append(nn.LSTM(hidden_dim, hidden_dim))
        self.attention = attention
        self.embed = nn.Embedding(num_class, hidden_dim)

        self.fc = nn.Linear(hidden_dim + context_dim, hidden_dim)
        self.activate = torch.nn.LeakyReLU(negative_slope=0.2)
        self.unembed = nn.Linear(hidden_dim, num_class)
        self.unembed.weight = self.embed.weight
        self.character_distribution = nn.Sequential(self.fc, self.activate, self.unembed)

    def forward(self, input, seq_sizes, max_iters):
        raw_pred_seq = []
        attention_record = []
        for step in range(max_iters):
            attention_score, raw_pred, state = self.forward_step(input, seq_sizes, output_word, state)
            attention_record.append(attention_score)
            raw_pred_seq.append(raw_pred)
            output_word = torch.max(raw_pred, dim=1)[1]
        return torch.stack(raw_pred_seq, dim=1), attention_record

    def forward_step(self, input, seq_sizes, output_word, state):
        output_word_emb = self.embed(output_word)
        hidden, cell = state[0], state[1]
        last_rnn_output = hidden[-1]
        attention_score, context = self.attention(last_rnn_output, input, seq_sizes)

        rnn_input = torch.cat([output_word_emb, context], dim=1)
        new_hidden, new_cell = [None] * len(self.rnn_layer), [None] * len(self.rnn_layer)
        for i, rnn in enumerate(self.rnn_layer):
            new_hidden[i], new_cell[i] = rnn(rnn_input, (hidden[i], cell[i]))
            rnn_input = new_hidden[i]
        rnn_output = new_hidden[-1]

        concat_feature = torch.cat([rnn_output, context], dim=1)
        raw_pred = self.character_distribution(concat_feature)
        return attention_score, raw_pred, [new_hidden, new_cell]


class Attention(nn.Module):
    def __init__(self, key_query_dim=128, speller_query_dim=256, listener_feature_dim=512, context_dim=128):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax()
        self.fc_query = nn.Linear(speller_query_dim, key_query_dim)
        self.fc_key = nn.Linear(listener_feature_dim, key_query_dim)
        self.fc_value = nn.Linear(listener_feature_dim, context_dim)
        self.activate = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, decoder_state, input, seq_sizes):
        query = self.activate(self.fc_query(decoder_state))

        batch_size = input.size(0)
        time_steps = input.size(1)
        reshaped_x = input.contiguous().view(-1, input.size(-1))
        output_x = self.fc_key(reshaped_x)
        key = self.activate(output_x.view(batch_size, time_steps, -1))

        energy = torch.bmm(query.unsqueeze(1), key.transpose(1, 2)).squeeze(dim=1)

        mask = Variable(energy.data.new(energy.size(0), energy.size(1)).zero_(), requires_grad=False)
        for i, size in enumerate(seq_sizes):
            mask[i, :size] = 1
        attention_score = mask * self.softmax(energy)
        attention_score /= torch.sum(attention_score, dim=1).unsqueeze(1).expand_as(attention_score)

        value = self.activate(self.fc_value(input))
        context = torch.bmm(attention_score.unsqueeze(1), value).squeeze(dim=1)

        return attention_score, context


class LAS(nn.Module):
    def __init__(self):
        super(LAS, self).__init__()
        self.listener = Listener(input_dim=40, hidden_dim=256)
        self.attention = Attention(key_query_dim=128, speller_query_dim=256, listener_feature_dim=512, context_dim=128)
        self.speller = Speller(hidden_dim=256, context_dim=128, rnn_layer=3, attention=self.attention, num_class=33)

    def forward(self, input, seq_sizes, labels, max_iters=250):
        listener_features, out_seq_sizes = self.listener(input, seq_sizes)
        outputs, attentions = self.speller(listener_features, out_seq_sizes, max_iters)
        return outputs, attentions

