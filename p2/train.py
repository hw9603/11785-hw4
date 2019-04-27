import time
import numpy as np
import random
import torch
import torch.nn as nn
from dataloader import get_loaders
from config import Config
from model import Listener, Speller
from utils import ER, calculate_loss


def train(train_loader, dev_loader, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, e, teacher_forcing_ratio=0.9):
    encoder.train()
    decoder.train()
    encoder.to(Config.DEVICE)
    decoder.to(Config.DEVICE)

    avg_loss = 0.0
    t = time.time()
    print("epoch", e)
    epoch_loss = 0
    for batch_idx, (data_batch, label_batch, input_lengths, target_lengths) in enumerate(train_loader):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, hidden = encoder(data_batch, input_lengths)
        decoder_outputs = decoder(encoder_outputs, teacher_forcing_ratio, label_batch)

        loss = calculate_loss(decoder_outputs, label_batch, target_lengths, criterion)
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        epoch_loss += loss.item()
        avg_loss += np.exp(loss.item())
        if batch_idx % Config.LOG_INTERVAL == Config.LOG_INTERVAL - 1:
            print("[Train Epoch %d] batch_idx=%d [%.2f%%, time: %.2f min], loss=%.4f" %
                  (e, batch_idx, 100. * batch_idx / len(train_loader), (time.time() - t) / 60,
                   avg_loss / Config.LOG_INTERVAL))
            avg_loss = 0.0

    print("Loss: {}".format(epoch_loss / len(train_loader)))


def eval(loader, encoder, decoder):
    encoder.eval()
    decoder.eval()
    encoder.to(Config.DEVICE)
    decoder.to(Config.DEVICE)

    error = 0
    error_rate_op = ER()
    for batch_idx, (data_batch, label_batch, input_lengths, target_lengths) in enumerate(loader):
        encoder_outputs, hidden = encoder(data_batch)
        decoder_outputs, hidden = decoder(encoder_outputs, hidden)
        error += error_rate_op(decoder_outputs, input_lengths, label_batch)
    print("total error: ", error / loader.dataset.total_chars)
    return error / loader.dataset.total_chars


def main():
    print(Config.DEVICE)
    train_loader, dev_loader, test_loader = get_loaders()
    encoder = Listener(input_dim=Config.INPUT_DIM, hidden_dim=Config.LISTENER_HIDDEN_SIZE)
    decoder = Speller(hidden_size=Config.SPELLER_HIDDEN_SIZE,
                      embed_size=Config.SPELLER_EMBED_SIZE,
                      context_size=0,  # TODO: this is for debug purpose (without attention)
                      output_size=Config.NUM_CLASS)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=Config.LR)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss(reduction='none')
    for e in range(Config.EPOCHS):
        train(train_loader, dev_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, e)
        eval(dev_loader, encoder, decoder)
        torch.save(encoder.state_dict(), "models/encoder" + str(e) + ".pt")
        torch.save(decoder.state_dict(), "models/decoder" + str(e) + ".pt")
    print("Done! Yeah~")


if __name__ == "__main__":
    main()
