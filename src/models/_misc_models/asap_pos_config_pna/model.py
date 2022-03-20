import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GraphConv, global_mean_pool,
                                JumpingKnowledge, PNAConv)
from src.base_model import BaseModel
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
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
    def __init__(self, nn: Callable, aggr: str = 'max', use_norm=False, **kwargs):
        super(CustomEdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()
        self.use_norm = use_norm

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, edge_attr=None) -> Tensor:
        """"""
        x = torch.cat((pos, x), dim=1)
        #if self.use_norm:
        #    x = torch.cat((norm, x), dim=1)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr.unsqueeze(1), size=None)


    def message(self, x_i: Tensor, x_j: Tensor, edge_attr) -> Tensor:
        if self.use_norm:
            # print("use norm?!")
            return self.nn(torch.cat([x_i[:, 0:3], edge_attr, x_j[:, 0:3] - x_i[:, 0:3], x_i[:, 3:]], dim=-1))
        # print("NOOOO use norm?!")
        return self.nn(torch.cat([x_i[:, 0:3], x_j[:, 0:3] - x_i[:, 0:3], x_i[:, 3:]], dim=-1))

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
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super(CustomEdgeConvWeighted, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

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
        if edge_weight is None:
            return self.nn(torch.cat([x_i[:, 0:3], x_j[:, 0:3] - x_i[:, 0:3], x_i[:, 3:]], dim=-1))
        return self.nn(torch.cat([x_i[:, 0:3], x_j[:, 0:3] - x_i[:, 0:3], x_i[:, 3:]], dim=-1)) * edge_weight.view(-1, 1)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, sel>f.nn)

class ASAP(torch.nn.Module):
    def __init__(self, num_classes, num_input_channels, num_layers=10, hidden=64, ratio=0.7, dropout=0.1, class_dropout=0.5, use_norm=False):
        super(ASAP, self).__init__()
        #self.conv1 = GraphConv(num_input_channels, hidden, aggr='mean')
        extra_input_neuros = 1 if use_norm else 0
        #self.conv1 = CustomEdgeConv(use_norm=use_norm, nn=MLP([2 * 3 + num_input_channels - 3 + extra_input_neuros, hidden, hidden, hidden]), aggr="max")
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.conv1 = PNAConv(in_channels=5, out_channels=hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=1, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            #GraphConv(hidden, hidden, aggr='mean')
            CustomEdgeConvWeighted(nn=MLP([2 * 3 + hidden, hidden, hidden, hidden]), aggr="max")
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

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, pos, normal, edge_attr, edge_index, batch = data.x, data.pos, data.normal, data.edge_attr, data.edge_index, data.batch
        edge_weight = None
        x = F.relu(self.conv1(x, pos, edge_index, edge_attr))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, pos=pos, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, perm = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
                #print("perm size", perm.size())
                pos = pos[perm]
                #print("x size", x.size())
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.class_dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Model(BaseModel):
    def get_model(self, num_classes=9):
        num_input_channels = self.config.num_input_channels if hasattr(self.config, 'num_input_channels') else 8
        num_layers = self.config.asap_num_layers if hasattr(self.config, 'asap_num_layers') else 6
        hidden = self.config.asap_num_layers if hasattr(self.config, 'asap_hidden_neurons') else 64
        ratio = self.config.asap_num_layers if hasattr(self.config, 'asap_cluster_ratio') else 0.7
        dropout = self.config.asap_num_layers if hasattr(self.config, 'asap_dropout') else 0.1
        class_dropout = self.config.class_dropout if hasattr(self.config, 'asap_class_dropout') else 0.5
        use_norm = self.config.asap_use_norm if hasattr(self.config, 'asap_use_norm') else False

        self.model = ASAP(
            num_classes=num_classes,
            num_layers=num_layers,
            num_input_channels=num_input_channels,
            hidden=hidden,
            ratio=ratio,
            dropout=dropout,
            class_dropout=class_dropout,
            use_norm=use_norm,
        )
