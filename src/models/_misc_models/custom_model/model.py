import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import (ASAPooling, GraphConv, global_mean_pool,
                                JumpingKnowledge, NNConv)
from torch_geometric.nn.conv import MessagePassing
from src.base_model import BaseModel
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

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

class EdgeFeatureConv(MessagePassing):
    def __init__(self, num_edge_features, out_channels):
        super(EdgeFeatureConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).

        # 5 edge features + cosine simlarity of the vertice normals of i, j
        #in_channels = 5 + 1
        #self.lin = torch.nn.Linear(in_channels, out_channels)
        self.mlp = Seq(Lin(num_edge_features, 20), ReLU(), Lin(20, out_channels))

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, channels]

        #edge_attr = self.lin(edge_attr)
        edge_index = edge_index.type(torch.long)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x contains the norm
        # cosine is calcuated throug:
        #   v * w / |v| * |w|
        # print(x_i)
        # print(torch.norm(x_i, dim=1))
        # dot = x_i[:, 0] * x_j[:, 0] + x_i[:, 1] * x_j[:, 1] + x_i[:, 2] * x_j[:, 2]
        # cos = dot / (torch.norm(x_i, dim=1) * torch.norm(x_j, dim=1))
        # print("cos", cos)
        # print("cos ", cos.view(-1, 1).size())
        # print("edge attr size", edge_attr.size())
        msg = torch.cat((cos.view(-1, 1), edge_attr), dim=1)
        #print("msg size", msg.size())
        print("msg", msg.size())
        msg = msg.type(torch.float)

        # Step 4: Normalize node features.
        return self.mlp(msg)

class ASAP(torch.nn.Module):
    def __init__(self, num_classes, num_input_channels, num_edge_features, num_layers=10, hidden=64, ratio=0.9, dropout=0.1):
        super(ASAP, self).__init__()
        self.edge_conv = NConvEdgeFeature(num_edge_features=num_edge_features, out_channels=num_input_channels)
        self.conv1 = GraphConv(num_input_channels, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
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
        x = F.relu(self.conv1(x, edge_index))
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
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
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
            num_layers=7,
            dropout=0.0,
            ratio=0.25,
            hidden=16,)
