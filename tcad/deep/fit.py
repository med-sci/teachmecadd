import torch
from torch.nn import BCELoss
from torch.optim import SGD
from tqdm import tqdm


def train(model, device, data_loader):
    for _, batch in enumerate(data_loader):
        batch = batch.to(device)

    optimizer = SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    out = model(batch)
    loss = BCELoss()
    loss = loss(out, batch.y.float())
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, device, data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for _, batch in enumerate(data_loader):
            batch = batch.to(device)
            out = model(batch)

            for idx, pred in enumerate(out):

                if torch.round(pred) == batch.y[idx]:
                    correct += 1
                total += 1

        return round((correct / total * 100), 3)
