import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib_metadata import re
from numpy.core.numeric import indices
from sklearn import linear_model
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.linear import Linear
from torch_geometric.nn import BatchNorm, Set2Set
from torch_geometric.nn.conv import GATConv, GCNConv

from tcad.tools.nntools import get_atom_features_dims

ATOM_FEATURE_DIMS = get_atom_features_dims()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class GCN_Graph(nn.Module):
    def __init__(self, hidden_dim, encode_dim):
        super(GCN_Graph, self).__init__()

        self.node_encoder = AtomEncoder(hidden_dim, ATOM_FEATURE_DIMS)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.bn = BatchNorm(hidden_dim)

        self.pool = Set2Set(hidden_dim, processing_steps=4, num_layers=1)

        self.linear1 = torch.nn.Linear(hidden_dim * 2, encode_dim)
        self.linear2 = torch.nn.Linear(encode_dim, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, batched_data, return_embeds=False):
        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        embed = self.node_encoder(x)

        out = F.relu(
            self.conv1(
                embed,
                edge_index,
            )
        )
        out = self.bn(out)

        out = F.relu(
            self.conv2(
                out,
                edge_index,
            )
        )
        out = self.bn(out)

        out = self.conv3(
            out,
            edge_index,
        )

        out = self.pool(out, batch)

        if return_embeds:
            return self.linear1(out)

        out = self.linear1(out)
        out = self.linear2(out)

        return self.sigmoid(out)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 5, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.Flatten(start_dim=1),
            nn.Linear(7296, 2400),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2400, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid()
        )

    def forward(self, x, return_embeds=False):
        encoded = self.encoder(x)

        if return_embeds:
            return encoded

        out = self.classifier(encoded)

        return out


class CNNAutoEncoder(nn.Module):
    def __init__(self, desired_dim: int) -> None:
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
            nn.Linear(1200, desired_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(desired_dim, 1200),
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

    def forward(self, x, return_embedings=False):
        encoded = self.encoder(x)

        if return_embedings:
            return encoded

        decoded = self.decoder(encoded)
        return decoded


class VaeCnn(nn.Module):
    def __init__(self, desired_dim: int) -> None:
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
            nn.Linear(desired_dim, 1200),
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
            nn.Sigmoid(),
        )

    def reparameterize(self, z_mean, z_log_var):
        eps = torch.randn(z_mean.size(0), z_mean.size(1)).to(DEVICE)
        z = z_mean + eps * torch.exp(z_log_var / 2.0)
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
            nn.Linear(2304, 1),
        )

        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 2400),
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
            nn.Tanh(),
        )

    def discriminator_forward(self, smile):
        return self.discriminator(smile)

    def generator_forward(self, array):
        return self.generator(array)
