import time
import numpy as np
import random
import torch
import torch.nn as nn
from dataloader import get_loaders
from config import Config
from model import Listener, Speller, LAS
from utils import ER, calculate_loss, plot_grad_flow


def train(train_loader, dev_loader, model, optimizer, criterion, e, teacher_forcing_ratio=0.9):
    model.train()
    model.to(Config.DEVICE)

    avg_loss = 0.0
    t = time.time()
    print("epoch", e)
    epoch_loss = 0
    for batch_idx, (data_batch, label_batch, input_lengths, target_lengths) in enumerate(train_loader):
        optimizer.zero_grad()

        decoder_outputs = model(data_batch, label_batch, input_lengths, teacher_forcing_ratio)

        loss = calculate_loss(decoder_outputs, label_batch, target_lengths, criterion)
        loss.backward()
        # plot the gradient flow
        plot_grad_flow(model.named_parameters())
        optimizer.step()
        epoch_loss += np.exp(loss.item())
        avg_loss += np.exp(loss.item())
        if batch_idx % Config.LOG_INTERVAL == Config.LOG_INTERVAL - 1:
            print("[Train Epoch %d] batch_idx=%d [%.2f%%, time: %.2f min], loss=%.4f" %
                  (e, batch_idx, 100. * batch_idx / len(train_loader), (time.time() - t) / 60,
                   avg_loss / Config.LOG_INTERVAL))
            avg_loss = 0.0

    print("Loss: {}".format(epoch_loss / len(train_loader)))


def eval(loader, model, teacher_forcing_ratio=0.9):
    model.eval()
    model.to(Config.DEVICE)

    error = 0
    error_rate_op = ER()
    for batch_idx, (data_batch, label_batch, input_lengths, target_lengths) in enumerate(loader):
        decoder_outputs = model(data_batch, label_batch, input_lengths, teacher_forcing_ratio)
        error += error_rate_op(decoder_outputs, input_lengths, label_batch)
    print("total error: ", error / loader.dataset.total_chars)
    return error / loader.dataset.total_chars


def prediction(loader, model, output_file):
    model.eval()
    model.to(Config.DEVICE)

    error_rate_op = ER()
    fwrite = open(output_file, "w")
    fwrite.write("Id,Predicted\n")
    line = 0
    for batch_idx, (data_batch, _, input_lengths, _) in enumerate(loader):
        decoder_outputs = model(data_batch, None, input_lengths, 0)
        decode_strs = error_rate_op(decoder_outputs, input_lengths)
        for s in decode_strs:
            if line % Config.LOG_INTERVAL == 0:
                print(line, s)
            fwrite.write(str(line) + "," + s + "\n")
            line += 1
    return


def main():
    print(Config.DEVICE)
    train_loader, dev_loader, test_loader = get_loaders()
    model = LAS()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    model.load_state_dict(torch.load("models/LAS_1.pt"))
    # prediction(test_loader, model, "prediction.csv")
    criterion = nn.CrossEntropyLoss(reduction='none')
    for e in range(Config.EPOCHS):
        train(train_loader, dev_loader, model, optimizer, criterion, e)
        torch.save(model.state_dict(), "models/LAS" + str(e) + ".pt")
        eval(dev_loader, model)
    print("Done! Yeah~")


if __name__ == "__main__":
    main()
