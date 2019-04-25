import torch
import torch.nn.functional as F
import Levenshtein as L
from character_list import CHARACTER_LIST


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
