import torch
import torch.nn.functional as F


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


def train_vae(dataloader, model, optimizer, epochs=5):
    losses = []

    for epoch in range(epochs):
        
        for idx, batch in enumerate(dataloader):
            encoded, z_mean, z_log_var, decoded = model(batch.to(DEVICE))
            
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                            - z_mean**2 
                                            - torch.exp(z_log_var), 
                                            axis=1)
            
            batchsize = kl_div.size(0)
            mse_loss = F.binary_cross_entropy(decoded, batch.to(DEVICE), reduction='none')
            mse_loss = mse_loss.view(batchsize, -1).sum(axis=1)

            loss = mse_loss.mean() + kl_div.mean()

            optimizer.zero_grad()

            loss.backward()
            losses.append(loss.item())
            
            optimizer.step()
        
        if epoch % 10 ==0:
            print(f"Epoch:{epoch} criterion_loss:{round(mse_loss.mean().item(),5)} KL_loss:{round(kl_div.mean().item(),5)} total_loss: {round(loss.item(),5)}")
    
    return losses


def train_gan(dataloader, model, optimizer_gen, optimizer_discr, epochs=5):
    gen_losses = []
    discr_losses = []
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):

        for batch in dataloader:

            batch_size = batch.shape[0]
            
            real_images = batch.to(DEVICE)
            real_labels = torch.ones(batch_size, device=DEVICE)
            
            noise = torch.randn(batch_size, model.latent_dim, device=DEVICE)

            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size, device=DEVICE)
            flipped_fake_labels = real_labels

            #train discriminator
            optimizer_discr.zero_grad()

            discr_pred_real = model.discriminator_forward(real_images).view(-1)
            real_loss = criterion(discr_pred_real, real_labels)

            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = criterion(discr_pred_fake, fake_labels)

            discr_loss = 0.5*(real_loss+fake_loss)
            discr_losses.append(discr_loss)
            discr_loss.backward()

            optimizer_discr.step()

            #train generator
            optimizer_gen.zero_grad()
            
            discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
            
            gener_loss = criterion(discr_pred_fake, flipped_fake_labels)
            gen_losses.append(gener_loss)
            gener_loss.backward()

            optimizer_gen.step()
        
        if epoch % 10 ==0:
            print(f"Epoch: {epoch} Discriminator loss: {discr_loss} Generator loss: {gener_loss}")
        









