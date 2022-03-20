import torch
import numpy as np
from heapq import heappop, heapify
from torch import nn
import open3d as o3d
from torch_geometric.nn import fps
from torch_scatter import scatter_max


class ClusterPoolFDS(nn.Module):

    def __init__(self, target=0, multi_thread=False):
        super(ClusterPool, self).__init__()

    def __call__(self, x, edge_index, target_edge_count, batch, vertices, edges):
        return self.forward(x, edge_index, target_edge_count, batch, vertices, edges)

    def forward(self, x, edge_index, num_clusters, batch, vertices, edges):
        for k in range(num_clusters):
            pass

class ClusterPoolHighest(nn.Module):

    def __init__(self, target=0, multi_thread=False):
        super(ClusterPool, self).__init__()

    def __call__(self, x, edge_index, target_edge_count, batch, vertices, edges):
        return self.forward(x, edge_index, target_edge_count, batch, vertices, edges)

    def forward(self, x, edge_index, num_clusters, batch, vertices, edges):
        x_norm = torch.sum(x * x, 1)

        neighbours = {}
        for i in range(len(edge_index[0])):
            from_n = int(edge_index[0][i])
            to_n = int(edge_index[1][i])
            if from_n not in neighbours:
                neighbours[from_n] = []
            neighbours[from_n].append(to_n)

        for batch_id in range(batch.max().item() + 1):
            batch_node_ids = (batch == batch_id).nonzero().view(-1)
            batch_x_norm = x_norm[batch_node_ids]
            batch_vertice_norm = vertices[batch_node_ids]

            centroids = torch.zeros(num_clusters)
            centroid_ids = torch.zeros(num_clusters)
            new_edge_index = {}

            for k in range(num_clusters):
                #out, argmax = scatter_max(x_norm, batch)
                centroid_id = torch.argmax(batch_x_norm)
                centroid_ids[k] = centroid_id
                centroids[k] = vertices[k]
                # batch_x_norm reduce norm of the points around the new centroids just slightly
                new_edge_index[k] = []  # + offset for the batch

            distances = vertices.repeat(1, num_clusters, 1).subtract(centroids).matmul(centroids).norm()
            # for each row get the centroids id (argmin for each row)
            centroid_alloc = torch.argmax(distances, dim=1)
            # calculate new centroid feature by global mean pooling for all members
            #             x_norm_cl =
            #                 torch.randn(4,2)
            # ids = torch.Tensor([1,1,0,0]).long()
            # print(m.gather(1, ids.view(-1,1)))
            new_centroid_features = scatter(batch_x_norm, centroid_alloc, dim=1, reduce="mean")
            # create new edge index by going through all members, and their neighbours, if the neighbours belong to another
            #batch_x_norm[centroid_ids]
            already_connected = set()
            for node_id in batch_node_ids:
                for nb in neighbours[node_id]:
                    if nb in already_connected:
                        continue
                    # TODO: consider offset of the batch for node_id..
                    if centroid_alloc[node_id] == centroid_alloc[nb]:
                        continue
                    from_k = centroid_alloc[node_id]
                    to_k = centroid_alloc[nb]
                    if to_k not in new_edge_index[from_k]:
                        new_edge_index[from_k].append(to_k)
                        new_edge_index[to_k].append(from_k)

            # cluster, add a link between them

            # TODO: merge, x features and edge_index, and vertices





def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N  # custom change, added - 1, see https://github.com/yanx27/Pointnet_Pointnet2_pytorch/issues/54
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N  # custom change, added - 1, see https://github.com/yanx27/Pointnet_Pointnet2_pytorch/issues/54
    group_idx[mask] = group_first[mask]
    return group_idx
