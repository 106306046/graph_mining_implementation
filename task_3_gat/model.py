import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv


class GCN(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """

    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(GraphConv(in_size, hid_size, activation=F.relu_))
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, features):
        h = features
        print(h.size())
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        print(h)
        return h


class YourGNNModel(nn.Module):
    """
    TODO: Use GCN model as reference, implement your own model here to achieve higher accuracy on testing data
    """

    def __init__(self, in_size, hid_size, out_size, num_heads):
        super().__init__()
        self.layer1 = GATConv(in_size, hid_size, num_heads)
        self.layer2 = GATConv(hid_size * num_heads, out_size, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = h.view(-1, h.size(1) * h.size(2))
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(g, h)
        h = h.squeeze()
        return h
