import torch


class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 24
    DEV_BATCH_SIZE = 16
    LOG_INTERVAL = 20
    LR = 1e-4
    WDECAY = 1e-5
    EPOCHS = 10

    INPUT_DIM = 40
    LISTENER_HIDDEN_SIZE = 256
    SPELLER_HIDDEN_SIZE = 512  # change to 512
    SPELLER_EMBED_SIZE = 512
    CONTEXT_SIZE = 128
    KEY_QUERY_VAL_SIZE = 128

    NUM_CLASS = 33

    EOS = '<eos>'
