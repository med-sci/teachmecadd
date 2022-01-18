from numpy.core.numeric import indices
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.linear import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GCNConv

from tcad.tools.nntools import get_atom_features_dims

ATOM_FEATURE_DIMS = get_atom_features_dims()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, atom_feature_dims):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for _, dim in enumerate(atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0

        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class GCN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=True
    ):
        super(GCN, self).__init__()

        self.convs = self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)]
            + [
                GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
                for _ in range(num_layers - 2)
            ]
            + [GCNConv(in_channels=hidden_dim, out_channels=output_dim)]
        )
        self.bns = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(num_features=hidden_dim)
                for _ in range(num_layers - 1)
            ]
        )

        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        out = None

        for conv, bn in zip(self.convs[:-1], self.bns):
            x1 = F.relu(bn(conv(x, adj_t)))

            if self.training:
                x1 = F.dropout(x1, p=self.dropout)
            x = x1
        x = self.convs[-1](x, adj_t)
        out = x if self.return_embeds else self.sigmoid(x)

        return out


class GCN_Graph(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        self.node_encoder = AtomEncoder(hidden_dim, ATOM_FEATURE_DIMS)

        self.gnn_node = GCN(
            hidden_dim, hidden_dim, hidden_dim, num_layers, dropout, return_embeds=True
        )

        self.pool = global_mean_pool
        self.linear = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, batched_data, return_embeds=False):
        x, edge_index, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.batch,
        )
        embed = self.node_encoder(x)
        x = self.gnn_node(embed, edge_index)
        features = self.pool(x, batch)

        if return_embeds:
            return features
        out = self.linear(features)
        return self.sigmoid(out)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, 11, 1),
            nn.ReLU(),
            nn.MaxPool2d(5,stride=1),
            nn.Conv2d(128, 64, 11, 1),
            nn.ReLU(),
            nn.MaxPool2d(9,stride=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Linear(96, 1)
        )


    def forward(self, x, encode=False):
        encoded = self.encoder(x)
        encoded = F.adaptive_max_pool2d(encoded, output_size= 1)
        if encode:
            return encoded.squeeze()
        flatten=torch.flatten(encoded,1)
        out = self.classifier(flatten)
        return out


class CNNAutoEncoder(nn.Module):
    def __init__(self, desired_dim:int)->None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 5, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.Flatten(start_dim=1),
            nn.Linear(7296, 2400),
            nn.LeakyReLU(),
            nn.Linear(2400, 1200),
            nn.LeakyReLU(),
            nn.Linear(1200, desired_dim)
        )
        self.decoder = nn.Sequential(
           nn.Linear(desired_dim,1200),
            nn.LeakyReLU(),
            nn.Linear(1200, 2400),
            nn.LeakyReLU(),
            nn.Linear(2400, 7296),
            nn.Unflatten(1, (128, 19, 3)),
            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 16, 6, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2),
        )
    
    def forward(self, x, return_embedings = False):
        encoded = self.encoder(x)
        
        if return_embedings:
            return encoded
        
        decoded = self.decoder(encoded)
        return decoded


        
class VaeCnn(nn.Module):
    def __init__(self, desired_dim:int)->None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 5, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.Flatten(start_dim=1),
            nn.Linear(7296, 2400),
            nn.LeakyReLU(),
            nn.Linear(2400, 1200),
        )

        self.z_mean = torch.nn.Linear(1200, desired_dim)
        self.z_log_var = torch.nn.Linear(1200, desired_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(desired_dim,1200),
            nn.LeakyReLU(),
            nn.Linear(1200, 2400),
            nn.LeakyReLU(),
            nn.Linear(2400, 7296),
            nn.Unflatten(1, (128, 19, 3)),
            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 16, 6, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2),
            nn.Sigmoid()
        )

    def reparameterize(self, z_mean, z_log_var):
        eps = torch.randn(z_mean.size(0), z_mean.size(1)).to(DEVICE)
        z = z_mean + eps * torch.exp(z_log_var/2.) 
        return z
    
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded


class GAN(nn.Module):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 5, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 2),
            nn.Flatten(start_dim=1),
            nn.Linear(2304,1)
        )

        self.generator = nn.Sequential(
            nn.Linear(latent_dim,2400),
            nn.LeakyReLU(),
            nn.Linear(2400, 7296),
            nn.Unflatten(1, (128, 19, 3)),
            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 16, 6, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 2),
        ) 

    def discriminator_forward(self, smile):
        return self.discriminator(smile)

    def generator_forward(self, array):
        return self.generator(array)

    





