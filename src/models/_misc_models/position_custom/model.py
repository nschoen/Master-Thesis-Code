import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling, EdgeConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from src.base_model import BaseModel
from torch_geometric.nn.norm import BatchNorm
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
        #print("pos.size", pos.size())
        #print("x.size", x.size())
        x = torch.cat((pos, x), axis=1)
        #print("x.size", x.size())
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)


    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        #print("x_i", x_i.size())
        #print("x_j", x_j.size())
        return self.nn(torch.cat([x_i[:, 0:3], x_j[:, 0:3] - x_i[:, 0:3], x_i[:, 3:]], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)



class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()

        self.conv1 = CustomEdgeConv(nn=MLP([2 * 3 + 5, 32, 32, 32]), aggr="max")
        self.conv2 = CustomEdgeConv(nn=MLP([2 * 3 + 32, 64, 64, 64]), aggr="max")
        self.conv3 = CustomEdgeConv(nn=MLP([2 * 3 + 64, 64, 128, 128]), aggr="max")

        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 32)
        self.lin3 = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch

        #x = torch.cat((pos, x), axis=1)

        x = self.conv1(x, pos, edge_index)
        x = self.conv2(x, pos, edge_index)
        x = self.conv3(x, pos, edge_index)
        x = gmp(x, batch)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class Model(BaseModel):
    def get_model(self, num_classes=9):
        self.model = Net(num_classes=num_classes, num_features=8)


