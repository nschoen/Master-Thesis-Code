import torch
import numpy as np
from heapq import heappop, heapify
from torch import nn
import time


class CollapsePool(nn.Module):

    def __init__(self, target=0, multi_thread=False):
        super(CollapsePool, self).__init__()

    def __call__(self, x, pos, edge_index, target_edge_count, batch):
        return self.forward(x, pos, edge_index, target_edge_count, batch)

    def forward(self, x, pos, edge_index, target_edge_count, batch):
        # keep track of nodes that need to be masked
        #x_mask = np.ones(x.size(0), dtype=np.bool)
        x_magnitude = torch.sum(x * x, 1)

        start = time.time()
        masked = []

        for batch_id in range(batch.max().item() + 1):
            node_ids = (batch == batch_id).nonzero().view(-1)
            num_nodes = node_ids.size(0)
            queue = self.build_queue(x_magnitude, node_ids)
            i = 0

            # t = torch.Tensor([1, 2, 3])  # get index of from node where x = node_id
            # print((t == 2).nonzero())
            #
            # a = torch.Tensor([1, 0, -1])
            # a[a < 0] = 0
            # a
            # tensor([1., 0., 0.])
            #
            # remvoe elements
            # i = 2
            # T = torch.tensor([1,2,3,4,5])
            # T = torch.cat([T[0:i], T[i+1:]])
            # or just mask?

            # 1. find neighbours by using (t == 2).nonzero() to find the indexes from edges_index[0] and reveive neighbour id's from edges_index[1]
            # 2. delete all neighbours from edges_index[0] and edges_index[1] based on (t == 2).nonzero()
            # 3. everywhere where old_id is in edge_idnex[1], change to new id
            # 4. reduce id > old_id by 1

            # print("queue", queue)
            while num_nodes > target_edge_count:
                node_id = int(queue[i].item())
                num_nodes -= 1
                masked.append(node_id)

                from_index = (edge_index[0] == node_id).nonzero()
                neighbour_ids = edge_index[1][from_index]
                neighbour_ids = neighbour_ids[neighbour_ids != node_id]  # remove self-reference

                min_nb_id = neighbour_ids[torch.argmin(x_magnitude[neighbour_ids])]

                old_id = node_id
                new_id = min_nb_id


                # merge into min_nb_id
                x[new_id] = x[new_id] + x[old_id] / 2

                # merge pos
                pos[new_id] = pos[new_id] + pos[old_id] / 2

                # remove all edges going out of old_id
                #edge_index[0][from_index] = -1
                #edge_index = edge_index[:, edge_index[0] != -1]

                # change edges going out of old_id to new_id if they do not reference the same node
                # duplicate = []
                new_node_outgoing_idx = (edge_index[0] == new_id).nonzero()
                new_node_current_neighbours = edge_index[1][new_node_outgoing_idx]
                j = -1
                for nb in edge_index[1][from_index]:
                    j += 1
                    if nb == new_id:
                        # remove edge that goes from old_id to new_id
                        edge_index[0][from_index[j]] = -1
                        continue
                    if nb == old_id:
                        # remove self loop
                        edge_index[0][from_index[j]] = -1
                        continue
                    if nb in new_node_current_neighbours:
                        # remove if new node already has this neighbour
                        edge_index[0][from_index[j]] = -1
                        continue
                    edge_index[0][from_index[j]] = new_id

                edge_index = edge_index[:, edge_index[0] != -1]

                # change edges going into old_id, to new_id if it's not coming from
                ref_old_id = (edge_index[1] == old_id).nonzero()
                j = -1
                for from_nb in edge_index[0][ref_old_id]:
                    j += 1
                    if from_nb == new_id:
                        # remove edge from new_id to old_id
                        edge_index[0][ref_old_id[j]] = -1
                        continue
                    if from_nb == old_id:
                        # remove self loop although it should not exist at this point any more
                        edge_index[0][ref_old_id[j]] = -1
                        continue
                    if from_nb in new_node_current_neighbours:
                        # if new_id already has this neighbour, remove this edge
                        edge_index[0][ref_old_id[j]] = -1
                        continue
                    edge_index[1][ref_old_id[j]] = new_id
                # TODO: find a better way to prevent duplicate reference from node a to new_id node

                edge_index = edge_index[:, edge_index[0] != -1]

                # remove edge going from new_id to old_id
                #from_new_index = (edge_index[0] == new_id).nonzero()
                #nbs = edge_index[1][from_new_index]
                #idx = (nbs == old_id).nonzero()[0][0]
                #delete_idx = from_new_index[idx]
                #edge_index[0][delete_idx] = -1
                #edge_index = edge_index[:, edge_index[0] != -1]
                #edge_index[0][delete_idx] = -1
                #edge_index[0] = edge_index[0][edge_index[0] != -1]
                #edge_index[1][delete_idx] = -1
                #edge_index[1] = edge_index[1][edge_index[1] != -1]

                # point edges that go into old_id, to new_id + add edges going from new_id to the previous neighbours of old_id
                # to_old_id_array_idx = (edge_index[1] == old_id).nonzero()
                # to_new_id_array_idx = (edge_index[1] == old_id).nonzero()
                # new_node_current_neighbours = edge_index[0][to_new_id_array_idx]
                # new_node_new_neighbours = edge_index[0][to_old_id_array_idx]
                # j = 0
                # for el in new_node_new_neighbours:
                #     if el in new_node_current_neighbours:
                #         continue
                #     edge_index[1][to_old_id_array_idx[j]] = new_id
                #     edge_index[1][to_old_id_array_idx[j]] = new_id
                #     j += 1
                # # TODO: find a better way to prevent duplicate reference from node a to new_id node
                #edge_index[1][to_old_id_array_idx] = new_id

                # remove x and x_magnitude entry
                x = torch.cat([x[0:old_id], x[old_id + 1:]])
                x_magnitude = torch.cat([x_magnitude[0:old_id], x_magnitude[old_id + 1:]])

                # reduce edge_ids > old_id
                edge_index[0][edge_index[0] > old_id] -= 1
                edge_index[1][edge_index[1] > old_id] -= 1

                # reduce node_ids > old_id
                queue[queue > old_id] -= 1

                # remove batch entry
                batch = torch.cat([batch[0:old_id], batch[old_id + 1:]])

                # remove pos entry
                pos = torch.cat([pos[0:old_id], pos[old_id + 1:]])

                i += 1

        print("took", time.time() - start, "secondsS")

        return x, pos, edge_index, batch

    def build_queue(self, features_magnitude, batch_node_ids):
        mag = features_magnitude[batch_node_ids]
        idx = torch.argsort(mag, dim=0)
        return batch_node_ids[idx]