import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GraphConv, global_mean_pool,
                                JumpingKnowledge)
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
    def __init__(self, nn: Callable, aggr: str = 'max', use_norm=False, use_absolute_pos=True, normalize_direction=False,
                 use_direction_angle=False, use_abs_direction=False, **kwargs):
        super(CustomEdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()
        self.use_norm = use_norm
        self.use_absolute_pos = use_absolute_pos
        self.normalize_direction = normalize_direction
        self.use_abs_direction = use_abs_direction
        self.use_direction_angle = use_direction_angle

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, edge_attr=None) -> Tensor:
        """"""
        x = torch.cat((pos, x), dim=1)
        #if self.use_norm:
        #    x = torch.cat((norm, x), dim=1)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        if edge_attr != None:
            edge_attr = edge_attr.unsqueeze(1)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=None)


    def message(self, x_i: Tensor, x_j: Tensor, edge_attr) -> Tensor:
        input = x_i
        input = torch.cat([x_j - x_i, input], dim=-1)
        input = torch.cat([edge_attr, input], dim=-1)  # if self.use_norm:
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
    def __init__(self, nn: Callable, aggr: str = 'max', use_absolute_pos=True, normalize_direction=False,
                 use_direction=True, use_diff=False, **kwargs):
        super(CustomEdgeConvWeighted, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()
        self.use_absolute_pos = use_absolute_pos
        self.normalize_direction = normalize_direction
        self.use_direction = use_direction
        self.use_diff = use_diff

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, edge_weight: OptTensor) -> Tensor:
        """"""
        x = torch.cat((pos, x), dim=1)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        torch.cuda.empty_cache()
        #print("edge_index.shape", edge_index.shape)
        #print("pos.shape", pos.shape)
        #print("x.shape", x[0].shape)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

    propagate_type = {'x': PairTensor, 'edge_weight': OptTensor}

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # calculate directional vector
        #print("x_j.shape", x_j.shape)
        #print("x_i.shape", x_i.shape)
        #print("is nan xi", torch.sum(torch.isnan(x_i)))
        #print("is nan xj", torch.sum(torch.isnan(x_j)))
        input = x_i
        input = torch.cat([x_j - x_i, input], dim=-1)
        #print("input.shape", input.shape)
        output = self.nn(input)
        if edge_weight is not None:
            return output * edge_weight.view(-1, 1)
        return output

        #if edge_weight is None:
        #    return self.nn(torch.cat([x_i[:, 0:3], x_j[:, 0:3] - x_i[:, 0:3], x_i[:, 3:]], dim=-1))
        #return self.nn(torch.cat([x_i[:, 0:3], x_j[:, 0:3] - x_i[:, 0:3], x_i[:, 3:]], dim=-1)) * edge_weight.view(-1, 1)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class ASAP(torch.nn.Module):
    def __init__(self, num_classes, num_input_channels, num_layers=10, hidden=64, ratio=0.7, dropout=0.1, class_dropout=0.5,
                 use_norm=False, use_absolute_pos=True, normalize_direction=False, convs_use_direction=True, convs_use_diff=False,
                 first_conv_use_direction_angle=False, first_conv_use_abs_pos=True, first_conv_use_abs_direction=True, mlp_num_hidden_layers=3):
        super(ASAP, self).__init__()
        #self.conv1 = GraphConv(num_input_channels, hidden, aggr='mean')

        mlp_hidden_layers = [hidden for i in range(mlp_num_hidden_layers)]

        extra_input_neuros = 1 if use_norm else 0
        # extra_input_neuros += 1
        if first_conv_use_direction_angle:
            extra_input_neuros += 1
        if first_conv_use_abs_direction:
            extra_input_neuros += 3
        if first_conv_use_abs_pos:
            extra_input_neuros += 3
        self.conv1 = CustomEdgeConv(use_norm=use_norm, nn=MLP([num_input_channels - 3 + extra_input_neuros] + mlp_hidden_layers),
                                    aggr="max",
                                    use_direction_angle=first_conv_use_direction_angle,
                                    use_absolute_pos=first_conv_use_abs_pos,
                                    use_abs_direction=first_conv_use_abs_direction,
                                    )
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        convs_extra_inputs = 0
        if convs_use_direction:
            convs_extra_inputs += 3
        if use_absolute_pos:
            convs_extra_inputs += 3
        if convs_use_diff:
            convs_extra_inputs += hidden
        self.convs.extend([
            #GraphConv(hidden, hidden, aggr='mean')
            CustomEdgeConvWeighted(nn=MLP([convs_extra_inputs + hidden] + mlp_hidden_layers), aggr="max",
                                   use_direction=convs_use_direction, use_diff=convs_use_diff,
                                   use_absolute_pos=use_absolute_pos, normalize_direction=normalize_direction)
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
        x, pos, edge_attr, edge_index, batch = data.x, data.pos, data.edge_attr, data.edge_index, data.batch
        edge_weight = None
        x = F.relu(self.conv1(x, pos, edge_index, edge_attr))
        if hasattr(self, 'save_features'):
            self.save_features(x, pos)
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, pos=pos, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            if hasattr(self, 'save_features'):
                #if i == len(self.convs) - 1:
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


class Model(BaseModel):
    def get_model(self, num_classes=9):
        num_input_channels = self.config.num_input_channels if hasattr(self.config, 'num_input_channels') else 8
        num_layers = self.config.asap_num_layers if hasattr(self.config, 'asap_num_layers') else 6
        hidden = self.config.asap_hidden_neurons if hasattr(self.config, 'asap_hidden_neurons') else 64
        ratio = self.config.asap_cluster_ratio if hasattr(self.config, 'asap_cluster_ratio') else 0.7
        dropout = self.config.asap_dropout if hasattr(self.config, 'asap_dropout') else 0.1
        class_dropout = self.config.asap_class_dropout if hasattr(self.config, 'asap_class_dropout') else 0.5
        use_norm = self.config.asap_use_norm if hasattr(self.config, 'asap_use_norm') else False
        use_absolute_pos = self.config.asap_use_absolute_pos if hasattr(self.config, 'asap_use_absolute_pos') else True
        normalize_direction = self.config.asap_normalize_direction if hasattr(self.config, 'asap_normalize_direction') else False
        convs_use_direction = self.config.asap_convs_use_direction if hasattr(self.config, 'asap_convs_use_direction') else True
        convs_use_diff = self.config.asap_convs_use_diff if hasattr(self.config, 'asap_convs_use_diff') else False
        first_conv_use_abs_pos = self.config.asap_first_conv_use_abs_pos if hasattr(self.config, 'asap_first_conv_use_abs_pos') else True
        first_conv_use_abs_direction = self.config.asap_first_conv_use_abs_direction if hasattr(self.config, 'asap_first_conv_use_abs_direction') else True
        first_conv_use_direction_angle = self.config.asap_first_conv_use_direction_angle if hasattr(self.config, 'asap_first_conv_use_direction_angle') else False
        mlp_num_hidden_layers = self.config.asap_mlp_num_hidden_layers if hasattr(self.config, 'asap_mlp_num_hidden_layers') else 3

        print("num_classes", num_classes)

        self.model = ASAP(
            num_classes=num_classes,
            num_layers=num_layers,
            num_input_channels=num_input_channels,
            hidden=hidden,
            ratio=ratio,
            dropout=dropout,
            class_dropout=class_dropout,
            use_norm=use_norm,
            use_absolute_pos=use_absolute_pos,
            normalize_direction=normalize_direction,
            convs_use_direction=convs_use_direction,
            convs_use_diff=convs_use_diff,
            first_conv_use_abs_pos=first_conv_use_abs_pos,
            first_conv_use_abs_direction=first_conv_use_abs_direction,
            first_conv_use_direction_angle=first_conv_use_direction_angle,
            mlp_num_hidden_layers=mlp_num_hidden_layers,
        )
