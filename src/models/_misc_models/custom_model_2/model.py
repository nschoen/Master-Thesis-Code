import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import (ASAPooling, GraphConv, global_mean_pool,
                                JumpingKnowledge, NNConv, EdgeConv, FeaStConv)
from torch_geometric.nn.conv import MessagePassing
from src.base_model import BaseModel
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ELU

from torch_scatter import scatter_add
from torch_geometric.utils import degree


class NConvEdgeFeature(torch.nn.Module):
    def __init__(self, num_edge_features, out_channels=32):
        super(NConvEdgeFeature, self).__init__()
        self.x_dim = 1
        nn1 = nn.Sequential(nn.Linear(num_edge_features, 25), nn.ReLU(), nn.Linear(25, self.x_dim * out_channels))
        self.conv = NNConv(self.x_dim, out_channels, nn1, aggr='mean')

    def forward(self, data):
        x_ones = torch.ones(data.x.size(0), self.x_dim).to(data.x.get_device() if data.x.get_device() != -1 else 'cpu')
        x = F.elu(self.conv(x_ones, data.edge_index, data.edge_attr))
        return x

def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ELU(), LayerNorm(channels[i]))
        for i in range(1, len(channels))
    ])


class LayerNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-5, affine=False):
        super(LayerNorm, self).__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(1, in_channels))
        self.bias = torch.nn.Parameter(torch.zeros(1, in_channels))
        if not affine:
            self.weight.requires_grad = False
            self.bias.requires_grad = False

    def forward(self, x, batch=None):
        """"""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        batch_size = batch.max().item() + 1

        count = degree(batch, batch_size, dtype=x.dtype).view(-1, 1).clamp(min=1) * x.shape[1]
        tmp = scatter_add(x, batch, dim=0, dim_size=batch_size)
        mean = tmp.sum(dim=1, keepdim=True) / count

        mean_diff = (x - mean[batch])
        tmp = scatter_add(mean_diff * mean_diff, batch, dim=0, dim_size=batch_size).sum(dim=1, keepdim=True)
        var = tmp / count

        out = (mean_diff / torch.sqrt(var[batch])) * self.weight + self.bias
        return out

class EdgeFeatureConv(MessagePassing):
    def __init__(self, num_edge_features, out_channels):
        super(EdgeFeatureConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).

        # 5 edge features + cosine simlarity of the vertice normals of i, j
        self.mlp = Seq(Lin(num_edge_features, 20), ELU(), Lin(20, out_channels))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, channels]

        edge_index = edge_index.type(torch.long)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        #msg = torch.cat((cos.view(-1, 1), edge_attr), dim=1)
        #msg = msg.type(torch.float)
        return self.mlp(edge_attr)

class ASAP(torch.nn.Module):
    def __init__(self, num_classes, num_input_channels, num_edge_features, num_layers=10, hidden=64, ratio=0.9, dropout=0.1):
        super(ASAP, self).__init__()
        #self.edge_conv = NConvEdgeFeature(num_edge_features=num_edge_features, out_channels=num_input_channels)
        self.edge_conv = EdgeFeatureConv(num_edge_features=num_edge_features, out_channels=num_input_channels)
        #self.conv1 = GraphConv(num_input_channels, hidden, aggr='mean')
        self.conv1 = EdgeConv(MLP([2 * num_input_channels, hidden, hidden, hidden]), 'add') #GraphConv(num_input_channels, hidden, aggr='mean')
        self.conv2 = FeaStConv(hidden, hidden, add_self_loops=True)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            #EdgeConv(MLP([2 * hidden, hidden, hidden]), 'mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(hidden, ratio, dropout=dropout)
            for i in range((num_layers) // 2)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        edge_index, batch, edge_attr = data.edge_index, data.batch, data.edge_attr
        x = self.edge_conv(data)
        edge_weight = None
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        #print("start loop")
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.elu(x)
            #print("x size", x.size())
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, _ = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
        #x = global_mean_pool(x, batch)
        #print(x.size())
        x = self.jump(xs)
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Model(BaseModel):
    def get_model(self, num_classes=9):
        # num_input_channels = 3
        # if self.config['mesh']['vertex_normals']:
        #     num_input_channels = 6

        data_item = self.test_dataset[0]
        num_edge_features = data_item.edge_attr.shape[1]
        self.model = ASAP(
            num_classes=num_classes,
            num_input_channels=16,
            num_edge_features=num_edge_features,
            num_layers=2,
            dropout=0.0,
            ratio=0.5,
            hidden=64,)