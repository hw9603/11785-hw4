import torch


class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 8
    LOG_INTERVAL = 1
    LR = 1e-3
    EPOCHS = 10

    INPUT_DIM = 40
    LISTENER_HIDDEN_SIZE = 256
    SPELLER_HIDDEN_SIZE = 512
    SPELLER_EMBED_SIZE = 256
    CONTEXT_SIZE = 128

    NUM_CLASS = 33

    EOS = '<eos>'
