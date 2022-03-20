import torch
import numpy as np
from heapq import heappop, heapify
from torch import nn


class CollapsePool(nn.Module):

    def __init__(self, target=0, multi_thread=False):
        super(CollapsePool, self).__init__()

    def __call__(self, x, edge_index, target_edge_count, batch, vertices, edges):
        return self.forward(x, edge_index, target_edge_count, batch, vertices, edges)

    def forward(self, x, edge_index, target_edge_count, batch, vertices, edges):
        collapse_mask = np.ones(x.size(0), dtype=np.bool)

        # keep track of nodes that need to be masked
        x_mask = np.ones(x.size(0), dtype=np.bool)

        neighbours = {}
        for i in range(len(edge_index[0])):
            from_n = int(edge_index[0][i])
            to_n = int(edge_index[1][i])
            if from_n not in neighbours:
                neighbours[from_n] = []
            neighbours[from_n].append(to_n)

        for batch_id in range(batch.max().item() + 1):
            node_ids = (batch == batch_id).nonzero().view(-1)
            index_offset = node_ids.min()
            num_nodes = node_ids.size(0)
            queue = self.build_queue(x, node_ids)

            while num_nodes > target_edge_count:
                if len(queue) == 0:
                    print("queue is empty?!")
                    break

                value, node_id = heappop(queue)
                node_id = int(node_id)

                if collapse_mask[node_id]:
                    collapse_mask[node_id] = False
                    neighbour_ids = neighbours[node_id]
                    num_nodes -= 1  # for current edge

                    new_feature = x[neighbour_ids].sum() / len(neighbour_ids)

                    # Alternatively: collapse into smallest or biggest neighbours?

                    new_neighbour_ids = []
                    for nb in neighbour_ids:
                        x_mask[nb] = False  # delete neighbhour
                        collapse_mask[nb] = False  # prevent selection from queue
                        num_nodes -= 1  # for current edge
                        for nbnb in neighbours[nb]:
                            if nbnb in neighbour_ids or nbnb == node_id:
                                continue
                            new_neighbour_ids.append(nbnb)
                            neighbours[nbnb] = [n for n in neighbours[nbnb] if n != nb]
                            neighbours[nbnb].append(node_id)

                    neighbours[node_id] = new_neighbour_ids

        x = x[torch.tensor(x_mask)]
        x_id_map_new_to_old = {}
        x_id_map_old_to_new = {}
        masked_count = 0
        for i, mask in enumerate(x_mask):
            if mask:
                x_id_map_old_to_new[i] = i - masked_count
                x_id_map_new_to_old[i - masked_count] = i
            else:
                masked_count += 1

        # create new edge_index
        new_edge_index = [[], []]

        for i in range(x.size(0)):
            new_x_id = i
            old_x_id = x_id_map_new_to_old[i]
            for nid in neighbours[old_x_id]:
                new_edge_index[0].append(new_x_id)
                new_edge_index[1].append(x_id_map_old_to_new[nid])

        batch = batch[torch.tensor(x_mask)]

        return x, torch.tensor(new_edge_index, dtype=torch.long).to(batch.device), batch

    def build_queue(self, features, batch_node_ids):
        squared_magnitude = torch.sum(features[batch_node_ids] * features[batch_node_ids], 1)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        heap = torch.cat((squared_magnitude, batch_node_ids.unsqueeze(-1)), dim=-1).tolist()
        heapify(heap)
        return heap