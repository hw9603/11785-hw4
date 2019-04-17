import torch


class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EOS = '<eos>'
    BATCH_SIZE = 64
