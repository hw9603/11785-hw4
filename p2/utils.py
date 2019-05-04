import numpy as np
import time
import torch
import torch.nn.functional as F
import Levenshtein as L
from character_list import CHARACTER_LIST
from config import Config
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


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
        # TODO: potential bug: preds shorter than trues, different mask lengths
        mask = torch.tensor([idx < seq_length[i] for idx in range(true.shape[0])]).to(Config.DEVICE)
        loss = criterion(pred, true)  # tensor
        loss = torch.where(mask > 0, loss, mask.float())
        # sum over sequence
        batch_loss = torch.sum(loss)
        avg_batch_loss += batch_loss
    # avg_batch_loss /= sum(seq_length)
    return avg_batch_loss


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.2)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("plots/grad_flow.png", bbox_inches='tight')
    plt.close()


def plot_attention(attentions):
    for attention_weights in attentions:
        fig = plt.figure()
        plt.imshow(attention_weights.cpu().detach().numpy())
        fig.savefig("plots/attentions/attention-{}.png".format(time.time()))
        plt.close()


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
                # print("PER:", ls * 100 / sum(len(s) for s in true))
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
