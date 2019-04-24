import time
import torch
import torch.nn as nn
from dataloader import get_loaders
from config import Config
from model import LAS


def train(train_loader, dev_loader, model, criterion, optimizer, e):
    model.train()
    model.to(Config.DEVICE)

    avg_loss = 0.0
    t = time.time()
    print("epoch", e)
    epoch_loss = 0
    for batch_idx, (data_batch, label_batch, input_lengths, target_lengths) in enumerate(train_loader):
        optimizer.zero_grad()
        # data_batch = data_batch.to(DEVICE)
        logits = model(data_batch, input_lengths)
        # print(logits)  # shape: [max_seq_len, batch_size, 47]
        loss = criterion(logits, label_batch)
        loss.mean().backward()
        optimizer.step()
        epoch_loss += loss.item()
        avg_loss += loss.item()
        if batch_idx % Config.LOG_INTERVAL == Config.LOG_INTERVAL - 1:
            print("[Train Epoch %d] batch_idx=%d [%.2f%%, time: %.2f min], loss=%.4f" %
                  (e, batch_idx, 100. * batch_idx / len(train_loader), (time.time() - t) / 60,
                   avg_loss / Config.LOG_INTERVAL))
            avg_loss = 0.0

    print("Loss: {}".format(epoch_loss / len(train_loader)))


def eval(loader, model):
    model.eval()
    model.to(Config.DEVICE)
    # error_rate_op = ER()
    error = 0
    for data_batch, labels_batch, input_lengths, target_lengths in loader:
        # data_batch = data_batch.to(DEVICE)
        predictions_batch = model(data_batch, input_lengths)
        # error += error_rate_op(predictions_batch, input_lengths, labels_batch)
    # print("total error: ", error / loader.dataset.total_phonemes)
    # return error / loader.dataset.total_phonemes


def main():
    train_loader, dev_loader, test_loader = get_loaders()
    model = LAS()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()
    for e in range(Config.EPOCHS):
        train(train_loader, dev_loader, model, criterion, optimizer, e)
        # per = eval(dev_loader, model)
        torch.save(model.state_dict(), "models/LAS" + str(e) + ".pt")
    print("Done! Yeah~")


if __name__ == "__main__":
    main()
