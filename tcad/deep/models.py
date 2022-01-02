import torch
import torch.nn as nn
import torch_geometric.nn.functional as F
from torch_geometric.nn.conv import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        super(GCN, self).__init__()
        
        self.convs = self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)                             
                for _ in range(num_layers-2)] + 
            [GCNConv(in_channels=hidden_dim, out_channels=output_dim)]    
        )
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=hidden_dim) for _ in range(num_layers-1)])
        
        self.softmax = torch.nn.LogSoftmax()
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
        out = x if self.return_embeds else self.softmax(x)

        return out