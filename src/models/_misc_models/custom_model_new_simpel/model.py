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



class ASAP(torch.nn.Module):
    def __init__(self, num_classes, num_input_channels, num_layers=10, hidden=64, ratio=0.9, dropout=0.1):
        super(ASAP, self).__init__()

        self.conv1 = FeaStConv(num_input_channels, int(hidden / 2), add_self_loops=True)
        self.conv2 = FeaStConv(int(hidden / 2), hidden, add_self_loops=True)
        self.conv3 = FeaStConv(hidden, hidden, add_self_loops=True)
        #self.conv1 = EdgeConv(MLP([2 * num_input_channels, hidden, hidden]), 'mean') #GraphConv(num_input_channels, hidden, aggr='mean')
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
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        #x = self.edge_conv(data)
        edge_weight = None
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        #print("start loop")
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
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
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0, training=self.training)
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
        #num_edge_features = data_item.edge_attr.shape[1]
        num_input_channels = data_item.x.shape[1]
        self.model = ASAP(
            num_classes=num_classes,
            num_input_channels=num_input_channels,
            num_layers=2,
            dropout=0.0,
            ratio=0.2,
            hidden=64,)
