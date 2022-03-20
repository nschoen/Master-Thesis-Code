import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from src.base_model import BaseModel
from pointnet2_classification import MLP


class Net(torch.nn.Module):
    def __init__(self, out_channels, normals=False, num_x_features=0, k=20, aggr='max'):
        super().__init__()
        self.normals = normals
        self.num_x_features = num_x_features
        input_features = 6 if normals else 3
        input_features += num_x_features
        self.conv1 = DynamicEdgeConv(MLP([2 * input_features, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x = pos
        if self.normals:
            x = torch.cat((x, data.normal), axis=-1)
        if self.num_x_features > 0:
            x = torch.cat((x, data.x), axis=-1)
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


class Model(BaseModel):
    def get_model(self, num_classes=9):
        normals = hasattr(self.config.dataset, 'point_normals') and self.config.dataset.point_normals
        num_x_features = self.config.dgcnn_num_x_features if hasattr(self.config, 'dgcnn_num_x_features') else 0
        print("normals", normals)
        self.model = Net(num_classes, normals=normals, k=20, num_x_features=num_x_features)
