import torch
from torch import Tensor
import torch.nn.functional as F
#from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch.nn import Linear
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, global_mean_pool, JumpingKnowledge, GraphConv
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
from src.base_model import BaseModel
from pointnet2_classification import MLP
from torch import Tensor
from torch_cluster import knn
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Union, Optional
from torch_geometric.nn.inits import reset
from asap_pooling import ASAPooling


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class CustomEdgeConv(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: Callable, aggr: str = 'max', use_norm=False, use_absolute_pos=True, normalize_direction=False, **kwargs):
        super(CustomEdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()
        self.use_norm = use_norm
        self.use_absolute_pos = use_absolute_pos
        self.normalize_direction = normalize_direction

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, pos: Tensor, edge_index: Adj, normal=None) -> Tensor:
        """"""
        if normal is not None:
            pos = torch.cat((pos, normal), dim=1)
        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)
        if isinstance(normal, Tensor):
            normal: PairTensor = (normal, normal)
        return self.propagate(edge_index=edge_index, x=pos, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        # calculate directional vector
        direction = x_j[:, 0:3] - x_i[:, 0:3]
        if self.normalize_direction:
            direction /= torch.norm(direction)

        input = torch.cat([direction, x_i[:, 0:3]], dim=-1)
        if self.use_norm:
            #normal_angle = np.arccos(np.clip(np.dot(edge_normals[edge_from_id], edge_normals[edge_to_id]), -1.0, 1.0))
            normal_angle = torch.acos(torch.clamp(torch.sum(x_i[:, 3:] * x_j[:, 3:], dim=-1), -1.0, 1.0))
            #print("normal_angle", normal_angle)
            input = torch.cat([normal_angle.unsqueeze(-1), input], dim=-1)
        #print("direction", direction)
        #print("input", input)
        #print("input size", input.shape)
        return self.nn(input)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class CustomEdgeConvWeighted(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: Callable, aggr: str = 'max', use_absolute_pos=True, normalize_direction=False, **kwargs):
        super(CustomEdgeConvWeighted, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()
        self.use_absolute_pos = use_absolute_pos
        self.normalize_direction = normalize_direction

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, edge_weight: OptTensor) -> Tensor:
        """"""
        x = torch.cat((pos, x), dim=1)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

    propagate_type = {'x': PairTensor, 'edge_weight': OptTensor}

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # calculate directional vector
        direction = x_j[:, 0:3] - x_i[:, 0:3]
        if self.normalize_direction:
            direction /= torch.norm(direction)

        input = torch.cat([direction, x_j[:, 3:] - x_i[:, 3:], x_i[:, 3:]], dim=-1)
        #if self.use_absolute_pos:
        #    input = torch.cat([x_i[:, 0:3], input], dim=-1)
        #print("input.shape", input.shape)
        output = self.nn(input)
        if edge_weight is not None:
            return output * edge_weight.view(-1, 1)
        return output

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class DgcnnASAP(torch.nn.Module):
    def __init__(self, num_classes, num_layers=10, hidden=64, ratio=0.7, dropout=0.1, class_dropout=0.5,
                 use_norm=False, use_absolute_pos=True, normalize_direction=False, k=20, num_workers=4):
        super(DgcnnASAP, self).__init__()
        #self.conv1 = GraphConv(num_input_channels, hidden, aggr='mean')
        self.k = k
        extra_input_neuros = 1 if use_norm else 0
        self.conv1 = CustomEdgeConv(use_norm=use_norm, nn=MLP([2 * 3 + extra_input_neuros, hidden, hidden, hidden]), aggr="max")
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            #GraphConv(hidden, hidden, aggr='mean')
            CustomEdgeConvWeighted(nn=MLP([3 + hidden * 2, hidden, hidden, hidden]), aggr="max")
            for i in range(num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(hidden, ratio[i] if isinstance(ratio, list) else ratio, dropout=dropout)
            for i in range((num_layers) // 2)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.class_dropout = class_dropout
        self.num_workers = num_workers

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        pos, normal, batch = data.pos, data.normal, data.batch
        edge_weight = None

        pos_pair = (pos, pos)
        b = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])
        edge_index = knn(pos_pair[0], pos_pair[1], self.k, b[0], b[1],
                         num_workers=self.num_workers)
        #print("pos shape", pos.shape)
        #print("edge_index", edge_index.shape)
        x = F.relu(self.conv1(pos, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, pos=pos, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            if hasattr(self, 'save_features'):
                if i == len(self.convs) - 1:
                    self.save_features(x, pos)
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, perm = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
                pos = pos[perm]
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.class_dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


# class Net(torch.nn.Module):
#     def __init__(self, out_channels, k=20, aggr='max'):
#         super().__init__()
#
#         self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
#         self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
#         self.lin1 = MLP([128 + 64, 1024])
#
#         self.mlp = Seq(
#             MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
#             Lin(256, out_channels))
#
#     def forward(self, data):
#         pos, batch = data.pos, data.batch
#
#         pos_pair = (pos, pos)
#
#         b = (None, None)
#         if isinstance(batch, Tensor):
#             b = (batch, batch)
#         elif isinstance(batch, tuple):
#             assert batch is not None
#             b = (batch[0], batch[1])
#
#         edge_index = knn(pos[0], pos[1], self.k, b[0], b[1],
#                          num_workers=self.num_workers)
#
#         x1 = self.conv1(pos, batch)
#         x2 = self.conv2(x1, batch)
#         out = self.lin1(torch.cat([x1, x2], dim=1))
#         out = global_max_pool(out, batch)
#         out = self.mlp(out)
#         return F.log_softmax(out, dim=1)


# class Model(BaseModel):
#     def get_model(self, num_classes=9):
#         self.model = Net(num_classes, k=20)


class Model(BaseModel):
    def get_model(self, num_classes=9):
        #num_input_channels = self.config.num_input_channels if hasattr(self.config, 'num_input_channels') else 8
        num_layers = self.config.asap_num_layers if hasattr(self.config, 'asap_num_layers') else 3
        hidden = self.config.asap_hidden_neurons if hasattr(self.config, 'asap_hidden_neurons') else 64
        ratio = self.config.asap_cluster_ratio if hasattr(self.config, 'asap_cluster_ratio') else 0.8
        dropout = self.config.asap_dropout if hasattr(self.config, 'asap_dropout') else 0.1
        class_dropout = self.config.class_dropout if hasattr(self.config, 'asap_class_dropout') else 0.5
        use_norm = self.config.asap_use_norm if hasattr(self.config, 'asap_use_norm') else False
        use_absolute_pos = self.config.asap_use_absolute_pos if hasattr(self.config, 'asap_use_absolute_pos') else True
        normalize_direction = self.config.asap_normalize_direction if hasattr(self.config, 'asap_normalize_direction') else False
        k = self.config.dgcnn_k if hasattr(self.config, 'dgcnn_k') else 20
        num_workers = 8

        self.model = DgcnnASAP(
            num_classes=num_classes,
            num_layers=num_layers,
            #num_input_channels=num_input_channels,
            hidden=hidden,
            ratio=ratio,
            dropout=dropout,
            class_dropout=class_dropout,
            use_norm=use_norm,
            use_absolute_pos=use_absolute_pos,
            normalize_direction=normalize_direction,
            k=k,
            num_workers=num_workers,
        )
