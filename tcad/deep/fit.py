import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_GCN(model, data_loader, optimizer, criterion):
    
    for _, batch in enumerate(data_loader):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.float())
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


def train_CNN(model, data_loader, optimizer, criterion):
    losses = []
    for _, batch in enumerate(data_loader):
        smiles, labels = batch
        smiles.to(DEVICE)
        optimizer.zero_grad()
        out = model(smiles).to(DEVICE)
        loss = criterion(out, labels.to(DEVICE))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


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

def train_encoder(model, dataloader, optimizer, criterion):

    for batch in dataloader:
        out = model(batch.to(DEVICE))
        loss = criterion(out, batch.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


