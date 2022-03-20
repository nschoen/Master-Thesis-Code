import os.path as osp
from math import ceil
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader, DataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, DenseGCNConv, GCNConv



from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MessagePassing

# class EdgeConv(MessagePassing):
#     def __init__(self, F_in, F_out):
#         super(EdgeConv, self).__init__(aggr='max')
#         self.mlp = Seq(Lin(2 * F_in, F_out), ReLU(), Lin(F_out, F_out))
#
#     def forward(self, x, edge_index):
#         # x has shape [N, F_in]
#         # edge_index has shape [2, E]
#         return self.propagate(edge_index, x=x)  # shape [N, F_out]
#
#     def message(self, x_i, x_j):
#         # x_i has shape [E, F_in]
#         # x_j has shape [E, F_in]
#         edge_features = torch.cat([x_i, x_j - x_i], dim=1)  # shape [E, 2 * F_in]
#         return self.mlp(edge_features)  # shape [E, F_out]



class EdgeFeatureConv(MessagePassing):
    def __init__(self, out_channels):
        super(EdgeFeatureConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).

        # 5 edge features + cosine simlarity of the vertice normals of i, j
        in_channels = 5 + 1
        #self.lin = torch.nn.Linear(in_channels, out_channels)
        self.mlp = Seq(Lin(in_channels, out_channels), ReLU(), Lin(out_channels, out_channels))

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


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, add_loop=False, lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        #self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        #self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)#, normalize=normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        #self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.conv3 = GCNConv(hidden_channels, out_channels)#, normalize=normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x, batch):
        print("x.size()", x.size())
        print("batch", batch.size())
        print("batch 2", batch)
        #batch_size, num_nodes, num_channels = x.size()

        #x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        #x = x.view(batch_size, num_nodes, num_channels)
        print("xs ", x.size())
        return x

    def forward(self, x, edge_index, batch, mask=None):
        #batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, edge_index)), batch)  # somehow add_loop method argument is missing, self.add_loop)))
        print("asdsad", x1.size())
        x2 = self.bn(2, F.relu(self.conv2(x1, edge_index)), batch)  #, self.add_loop)))
        x3 = self.bn(3, F.relu(self.conv3(x2, edge_index)), batch)  #, self.add_loop)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self, max_nodes=150, n_input_dim=3, num_classes=9):
        super(Net, self).__init__()

        edge_out_channels = 16
        self.edge_conv = EdgeFeatureConv(out_channels=edge_out_channels)

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(edge_out_channels, 64, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(edge_out_channels, 64, 64, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        num_nodes = ceil(16)
        self.gnn2a_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2a_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        mask = None

        x = self.edge_conv(x, edge_index, edge_attr)
        print("x.size edge", x.size())

        s = self.gnn1_pool(x, edge_index, batch, mask)
        x = self.gnn1_embed(x, edge_index, batch, mask)

        print("x.s", x.size())
        print("ed", edge_attr.size())
        x, edge_index, l1, e1 = dense_diff_pool(x, edge_index, s, mask)

        s = self.gnn2_pool(x, edge_index, batch)
        x = self.gnn2_embed(x, edge_index, batch)

        x, edge_index, l2, e2 = dense_diff_pool(x, edge_index, s)

        s = self.gnn2a_pool(x, edge_index, batch)
        x = self.gnn2a_embed(x, edge_index, batch)

        x, edge_index, l2, e2 = dense_diff_pool(x, edge_index, s)

        x = self.gnn3_embed(x, edge_index, batch)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        self.meta = (l1 + l2, e1 + e2)

        return F.log_softmax(x, dim=-1)


def setup_custom_training(
        train_dataset=None,
        test_dataset=None,
        batch_size=32,
        max_nodes=150,):

    num_classes = len(test_dataset.classes)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_loader = DenseDataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # train_loader = DenseDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    n_input_dim = train_dataset.get(0).x.size(1)
    model = Net(max_nodes=max_nodes, n_input_dim=n_input_dim, num_classes=num_classes)

    return (model, train_loader, test_loader)
