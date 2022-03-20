import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from src.base_model import BaseModel
from torch_geometric.nn.norm import BatchNorm

class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()

        self.conv1 = GraphConv(num_features, 32)
        self.bn1 = BatchNorm(32)
        self.conv1b = GraphConv(32, 64)
        self.bn1b = BatchNorm(64)
        self.pool1 = TopKPooling(64, ratio=0.5)
        self.conv2 = GraphConv(64, 64)
        self.bn2 = BatchNorm(64)
        self.pool2 = TopKPooling(64, ratio=0.5)
        self.conv3 = GraphConv(64, 64)
        self.bn3 = BatchNorm(64)
        self.pool3 = TopKPooling(64, ratio=0.5)
        self.conv4 = GraphConv(64, 64)
        self.bn4 = BatchNorm(64)
        self.pool4 = TopKPooling(64, ratio=0.3)

        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 32)
        self.lin3 = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn1b(self.conv1b(x, edge_index)))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #x = x1 + x2 + x3
        x = x4 + x3 + x2 + x1
        #x = x3 + x2 * 0.5 + x1 * 0.2

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class Model(BaseModel):
    def get_model(self, num_classes=9):
        self.model = Net(num_classes=num_classes, num_features=5)


