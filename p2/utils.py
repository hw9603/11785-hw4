import numpy as np
import torch
import torch.nn.functional as F
import Levenshtein as L
from character_list import CHARACTER_LIST


def calculate_loss(preds, trues, seq_length, criterion):
    # preds shape: (batch, length, 33)
    # trues shape: (batch, length)
    batch_size = preds.shape[0]
    assert preds.shape[0] == trues.shape[0]

    avg_batch_loss = 0
    for i, pred in enumerate(preds):
        # pred shape: (length, 33)
        # true shape: (length)
        true = trues[i]
        mask = [idx < seq_length[i] for idx in range(true.shape[0])]
        loss = criterion(pred, true, reduce=False)  # TODO: it might not be a numpy array. check type
        loss = np.ma.compressed(np.ma.masked_where(mask == 0, loss))
        # sum over sequence
        batch_loss = np.sum(loss)
        avg_batch_loss += batch_loss
    avg_batch_loss /= batch_size
    return avg_batch_loss


class ER:
    def __init__(self):
        self.label_map = CHARACTER_LIST
        self.decoder = greedy_decoder

    def __call__(self, prediction, seq_size, target=None):
        return self.forward(prediction, seq_size, target)

    def forward(self, prediction, seq_size, target=None):
        prediction = torch.transpose(prediction, 0, 1)
        prediction = prediction.cpu()
        probs = F.softmax(prediction, dim=2)
        # output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=torch.IntTensor(seq_size))
        output, out_seq_len = self.decoder()

        pred = []
        for i in range(output.size(0)):
            pred.append("".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]]))

        if target is not None:
            true = []
            for t in target:
                true.append("".join(self.label_map[o] for o in t))

                ls = 0.
                for p, t in zip(pred, true):
                    ls += L.distance(p.replace(" ", ""), t.replace(" ", ""))
                print("PER:", ls * 100 / sum(len(s) for s in true))
                return ls
        else:
            return pred


def greedy_decoder():
    raise NotImplemented
