import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GraphConv, SAGPooling, global_mean_pool,
                                JumpingKnowledge, EdgeConv)
from src.base_model import BaseModel
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ELU
from torch_scatter import scatter_add
from torch_geometric.utils import degree

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

class SAGPool(torch.nn.Module):
    def __init__(self, num_input_features, num_classes, num_layers, hidden, ratio=0.8):
        super(SAGPool, self).__init__()
        #self.conv1 = GraphConv(num_input_features, hidden, aggr='mean')
        self.conv1 = EdgeConv(MLP([2 * num_input_features, hidden, hidden]), 'add') #GraphConv(num_input_channels, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            #GraphConv(hidden, hidden, aggr='mean')
            EdgeConv(MLP([2 * hidden, hidden]), 'mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [SAGPooling(hidden, ratio) for i in range((num_layers) // 2)])
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.elu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, _, _ = pool(x, edge_index,
                                                     batch=batch)
        x = self.jump(xs)
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Model(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)

    def get_model(self, num_classes=9):
        data_item = self.test_dataset[0]
        num_input_features = data_item.x.shape[1]

        self.model = SAGPool(num_input_features, num_classes, num_layers=3, hidden=64, ratio=0.6)