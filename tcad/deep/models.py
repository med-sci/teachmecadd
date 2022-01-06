import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GCNConv

from tcad.tools.nntools import get_atom_features_dims

ATOM_FEATURE_DIMS = get_atom_features_dims()


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

        self.conv1 = nn.Conv2d(1, 128, 11, 1)
        self.pool1 = nn.MaxPool2d(5,stride=1)

        self.conv2 = nn.Conv2d(128, 64, 11, 1)
        self.pool2 = nn.MaxPool2d(9,stride=1)

        self.fc1 = nn.Linear(17024, 96)
        self.fc2 = nn.Linear(96, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.relu(x))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        return out












