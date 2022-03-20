import os.path as osp
from math import ceil
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, DenseGCNConv

#max_nodes = 150


# class MyFilter(object):
#     def __call__(self, data):
#         return data.num_nodes <= max_nodes


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, add_loop=False, lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))  # somehow add_loop method argument is missing, self.add_loop)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))  #, self.add_loop)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))  #, self.add_loop)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self, max_nodes=150, n_input_dim=3, num_classes=9):
        super(Net, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(n_input_dim, 64, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(n_input_dim, 64, 64, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        num_nodes = ceil(9)
        self.gnn2a_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2a_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, adj, mask = data.x, data.adj, data.mask

        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        s = self.gnn2a_pool(x, adj)
        x = self.gnn2a_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        self.meta = (l1 + l2, e1 + e2)

        return F.log_softmax(x, dim=-1)


def setup_dense_differentiable_pooling_training(
        train_dataset=None,
        test_dataset=None,
        batch_size=32,
        max_nodes=150,):

    # train_dataset.set_transform(T.ToDense(max_nodes))
    # # train_dataset.pre_filter = MyFilter
    # test_dataset.set_transform(T.ToDense(max_nodes))
    # test_dataset.pre_filter = MyFilter

    num_classes = len(test_dataset.classes)

    test_loader = DenseDataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    # max_nodes = 0
    #
    # for i in range(len(test_dataset)):
    #     data = test_dataset.get(i)
    #     print("data.num_nodes", data.num_nodes)
    #     print("lada", len(data.x))
    #     if data.num_nodes > max_nodes:
    #         max_nodes = data.num_nodes
    # # num max triangles + 2?
    # #752
    # print("max_nodes", max_nodes)

    train_loader = DenseDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    n_input_dim = train_dataset.get(0).x.size(1)
    model = Net(max_nodes=max_nodes, n_input_dim=n_input_dim, num_classes=num_classes)

    # def forward_pass_func(model_t, data):
    #     output, _, _ = model_t(data.x, data.adj, data.mask)
    #     return output

    return (model, train_loader, test_loader)


# @torch.no_grad()
# def test(model, loader, device):
#     model.eval()
#     correct = 0
#
#     for data in loader:
#         data = data.to(device)
#         pred = model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
#         correct += pred.eq(data.y.view(-1)).sum().item()
#     return correct / len(loader.dataset)
