import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from GCN2Conv import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from src.base_model import BaseModel

# transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
#dataset = Planetoid(path, dataset, transform=transform)
#data = dataset[0]
#data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.

# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn2_cora.py

class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, data):
        x, adj_t = data.x, gcn_norm(data.adj_t)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


# optimizer = torch.optim.Adam([
#     dict(params=model.convs.parameters(), weight_decay=0.01),
#     dict(params=model.lins.parameters(), weight_decay=5e-4)
# ], lr=0.01)


# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.adj_t)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

class Model(BaseModel):
    def __init__(self, config):
        dataset_transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
        BaseModel.__init__(self, config, dataset_transform)

    def get_model(self, num_classes=9):
        # self.test_dataset.get(0)
        # self.model = Net(12, 2, num_classes)
        num_input_channels = 3
        if self.config['mesh']['vertex_normals']:
            num_input_channels = 6

        self.model = Net(num_input_channels, num_classes, hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
                         shared_weights=True, dropout=0.6)