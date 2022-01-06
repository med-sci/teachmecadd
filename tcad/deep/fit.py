import torch
from torch.nn import BCELoss
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import SGD
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_GCN(model, data_loader):
    for _, batch in enumerate(data_loader):
        batch = batch.to(DEVICE)
        optimizer = SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        out = model(batch)
        loss = BCELoss()
        loss = loss(out, batch.y.float())
        loss.backward()
        optimizer.step()

    return loss.item()


def evaluate_GCN(model, data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for _, batch in enumerate(data_loader):
            batch = batch.to(DEVICE)
            out = model(batch)

            for idx, pred in enumerate(out):

                if torch.round(pred) == batch.y[idx]:
                    correct += 1
                total += 1

    return round((correct / total * 100), 3)


def train_CNN(model, data_loader):
    for _, batch in enumerate(data_loader):
        smiles, labels = batch
        smiles.to(DEVICE)
        optimizer = SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        out = model(smiles)
        loss = BCEWithLogitsLoss()
        loss = loss(out, labels)
        loss.backward()
        optimizer.step()

    return loss.item()


def evaluate_CNN(model, data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for _, batch in enumerate(data_loader):
            smiles, labels = batch
            smiles.to(DEVICE)
            out = model(smiles)

            for idx, pred in enumerate(out):

                if torch.round(pred) == labels[idx]:
                    correct += 1
                total += 1

    return round((correct / total * 100), 3)