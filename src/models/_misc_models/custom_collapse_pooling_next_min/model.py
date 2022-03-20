import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (ASAPooling, GraphConv, global_mean_pool,
                                JumpingKnowledge)
from torch_geometric.nn.norm import BatchNorm
from src.base_model import BaseModel
from collapse_pooling import CollapsePool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
from torch_geometric.nn.inits import reset

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
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super(CustomEdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)


    def forward(self, x: Union[Tensor, PairTensor], pos, edge_index: Adj) -> Tensor:
        """"""
        x = torch.cat((pos, x), axis=1)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)


    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        direction = x_j[:, 0:3] - x_i[:, 0:3]
        direction = direction / torch.norm(direction)
        print("asd", direction.size(), x_i.size())
        return self.nn(torch.cat([direction, x_i[:, 3:]], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class ASAP(torch.nn.Module):
    def __init__(self, num_classes, num_input_channels, hidden=64):
        super(ASAP, self).__init__()
        #self.conv1 = GraphConv(num_input_channels, 16, aggr='mean')
        self.conv1 = CustomEdgeConv(nn=MLP([num_input_channels, hidden, hidden, hidden]), aggr="max")
        self.pool1 = CollapsePool()
        #self.conv2 = GraphConv(16, 32, aggr='mean')
        self.conv2 = CustomEdgeConv(nn=MLP([3 + hidden, hidden, hidden, hidden]), aggr="max")
        self.pool2 = CollapsePool()
        #self.conv3 = GraphConv(32, 64, aggr='mean')
        self.conv3 = CustomEdgeConv(nn=MLP([3 + hidden, hidden, hidden, hidden]), aggr="max")
        self.pool3 = CollapsePool()
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
        x, pos, edge_index, batch, vertices, edge = data.x, data.pos, data.edge_index, data.batch, data.vertices, data.edges

        x = self.conv1(x, pos, edge_index)
        x, edge_index, batch, x_mask = self.pool1(x, edge_index, 1800, batch)
        pos = pos[x_mask]
        x = self.conv2(x, pos, edge_index)
        x, edge_index, batch, x_mask = self.pool2(x, edge_index, 1400, batch)
        pos = pos[x_mask]
        x = self.conv3(x, pos, edge_index)
        x, edge_index, batch, x_mask = self.pool3(x, edge_index, 800, batch)

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Model(BaseModel):
    def get_model(self, num_classes=9):
        num_input_channels = 8
        num_layers = self.config.dataset.num_layers if hasattr(self.config.dataset, "num_layers") else 6
        ratio = self.config.dataset.ratio if hasattr(self.config.dataset, "ratio") else 0.9
        print("num_layers", num_layers, ratio)
        self.model = ASAP(num_classes=num_classes, num_input_channels=num_input_channels)
