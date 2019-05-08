import time
import numpy as np
import random
import torch
import torch.nn as nn
from dataloader import get_loaders
from config import Config
from model import Listener, Speller, LAS
from utils import ER, calculate_loss


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
        char_loss = loss / sum(target_lengths)
        optimizer.step()
        epoch_loss += np.exp(char_loss.item())
        avg_loss += np.exp(char_loss.item())
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
        decoder_outputs = model(data_batch,
                                label_batch if teacher_forcing_ratio != 0 else None,
                                input_lengths, teacher_forcing_ratio)

        error += error_rate_op(decoder_outputs, input_lengths, label_batch)
    print("total error: ", error * 100 / loader.dataset.total_chars)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WDECAY)
    # model.load_state_dict(torch.load("models/LAS3/LAS3_20.pt"))
    eval(dev_loader, model, teacher_forcing_ratio=0)
    # prediction(test_loader, model, "prediction_.csv")
    criterion = nn.CrossEntropyLoss(reduction='none')
    teacher_force = 0.9
    for e in range(Config.EPOCHS):
        print("--------teacher force: {}---------".format(teacher_force))
        train(train_loader, dev_loader, model, optimizer, criterion, e, teacher_forcing_ratio=teacher_force)
        if teacher_force > 0.7:
            teacher_force -= 0.01
        torch.save(model.state_dict(), "models/LAS3/LAS3_{}.pt".format(e))
        eval(dev_loader, model, teacher_forcing_ratio=0)
        # if e % 5 == 0:
        #     prediction(test_loader, model, "prediction_{}.csv".format(e+3))
    print("Done! Yeah~")


if __name__ == "__main__":
    main()
