import torch


class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EOS = '<eos>'
    BATCH_SIZE = 64
    LOG_INTERVAL = 1
    LR = 1e-3
    EPOCHS = 10
