import os.path as osp
import os
import sys
from src.base_model import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)

transform = T.Cartesian(cat=False)

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class Net(nn.Module):
    def __init__(self, num_vertex_features, num_edge_features, num_classes):
        super(Net, self).__init__()
        nn1 = nn.Sequential(nn.Linear(num_edge_features, 25), nn.ReLU(), nn.Linear(25, 32 * num_vertex_features))
        self.conv1 = NNConv(num_vertex_features, 32, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(3, 25), nn.ReLU(), nn.Linear(25, 2048))  # 32 * 64 = 2048
        self.conv2 = NNConv(32, 64, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)  # data.edge_attr.size(1) = 3 (num of x features), not 10!

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
#
# def train(epoch):
#     model.train()
#
#     if epoch == 16:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.001
#
#     if epoch == 26:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.0001
#
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         F.nll_loss(model(data), data.y).backward()
#         optimizer.step()


# def test():
#     model.eval()
#     correct = 0
#
#     for data in test_loader:
#         data = data.to(device)
#         pred = model(data).max(1)[1]
#         correct += pred.eq(data.y).sum().item()
#     return correct / len(test_dataset)


# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(ROOT_DIR, 'models'))

class Model(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config, None) # transform

    def get_model(self, num_classes=9):
        data_item = self.test_dataset[0]
        num_vertex_features = data_item.x.shape[1]
        num_edge_features = data_item.edge_attr.shape[1]
        print(num_vertex_features)
        print(num_edge_features)
        self.model = Net(
            num_vertex_features,
            num_edge_features,
            num_classes)

        # self.model = MODEL.get_model(num_classes,
        #                              normal_channel=False,
        #                              first_SA_n_sample_factor=self.config['first_SA_n_sample_factor'])
        # self.criterion = MODEL.get_loss().to(self.config['device'])

    # def loss_criterion(self, predicted, target, complete_output=None):
    #     return self.criterion(predicted, target.long(), complete_output[1])

    # def get_data(self, data):
    #     points, target = data
    #     points = points.data.numpy()
    #     points = provider.random_point_dropout(points)
    #     points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
    #     points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
    #     points = torch.tensor(points)
    #     target = target[:, 0]
    #
    #     points = points.transpose(2, 1)
    #     return points.to(self.config['device']), target.to(self.config['device'])
    #


