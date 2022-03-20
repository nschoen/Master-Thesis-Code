import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (ASAPooling, GraphConv, global_mean_pool,
                                JumpingKnowledge)
from torch_geometric.nn.norm import BatchNorm
from src.base_model import BaseModel
from edge_pooling import MeshPool

class ASAP(torch.nn.Module):
    def __init__(self, num_classes, num_input_channels, hidden=64):
        super(ASAP, self).__init__()
        self.conv1 = GraphConv(num_input_channels, 16, aggr='mean')
        self.bn1 = BatchNorm(16)
        self.pool1 = MeshPool()
        self.conv2 = GraphConv(16, 32, aggr='mean')
        self.bn2 = BatchNorm(32)
        self.pool2 = MeshPool()
        self.conv3 = GraphConv(32, 64, aggr='mean')
        self.bn3 = BatchNorm(64)
        self.pool3 = MeshPool()
        #self.conv4 = GraphConv(32, hidden, aggr='mean')
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch, vertices, edge = data.x, data.edge_index, data.batch, data.vertices, data.edges

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x, edge_index, batch = self.pool1(x, edge_index, 1800, batch, vertices, edge)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x, edge_index, batch = self.pool2(x, edge_index, 1500, batch, vertices, edge)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x, edge_index, batch = self.pool3(x, edge_index, 1000, batch, vertices, edge)


        # x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        # x = F.relu(self.conv4(x, edge_index))
        #
        # self.conv1 = GraphConv(num_input_channels, 16, aggr='mean')
        # self.bn1 = Norm.BatchNorm2d(16)
        # self.pool1 = MeshPool()
        # self.conv2 = GraphConv(16, 32, aggr='mean')
        # self.bn2 = Norm.BatchNorm2d(16)
        # self.pool2 = MeshPool()
        # self.conv3 = GraphConv(32, 32, aggr='mean')
        # self.bn3 = Norm.BatchNorm2d(16)
        # self.pool3 = MeshPool()

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Model(BaseModel):
    def get_model(self, num_classes=9):
        num_input_channels = 5
        num_layers = self.config.dataset.num_layers if hasattr(self.config.dataset, "num_layers") else 6
        ratio = self.config.dataset.ratio if hasattr(self.config.dataset, "ratio") else 0.9
        print("num_layers", num_layers, ratio)
        self.model = ASAP(num_classes=num_classes, num_input_channels=num_input_channels)
