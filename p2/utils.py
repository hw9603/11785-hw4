import numpy as np
import torch
import torch.nn.functional as F
import Levenshtein as L
from character_list import CHARACTER_LIST
from config import Config


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
        mask = torch.tensor([idx < seq_length[i] for idx in range(true.shape[0])]).to(Config.DEVICE)
        loss = criterion(pred, true)  # tensor
        loss = torch.where(mask > 0, loss, mask.float())
        # sum over sequence
        batch_loss = torch.sum(loss)
        avg_batch_loss += batch_loss
    avg_batch_loss /= sum(seq_length)
    return avg_batch_loss


class ER:
    def __init__(self):
        self.label_map = CHARACTER_LIST
        self.decoder = greedy_decoder

    def __call__(self, prediction, seq_size, target=None):
        return self.forward(prediction, seq_size, target)

    def forward(self, prediction, seq_size, target=None):
        # prediction shape: (batch, length, 33)
        # target shape: (batch, length)
        output = self.decoder(prediction)

        pred = []
        for i in range(len(output)):
            # output[i] shape: (length,)
            p = ""
            for o in output[i]:
                if o == CHARACTER_LIST.index(Config.EOS):
                    break
                p += self.label_map[o]
            pred.append(p)

        if target is not None:
            true = []
            for t in target:
                tr = ""
                for o in t:
                    if o == CHARACTER_LIST.index(Config.EOS):
                        break
                    tr += self.label_map[o]
                true.append(tr)

                ls = 0.
                for p, t in zip(pred, true):
                    ls += L.distance(p, t)
                print("PER:", ls * 100 / sum(len(s) for s in true))
                return ls
        else:
            return pred


def greedy_decoder(prediction):
    # prediction shape: (batch, length, 33)
    output = []
    for i, p in enumerate(prediction):
        # p shape: (length, 33)
        output.append(torch.max(p, dim=1)[1])
    return output
