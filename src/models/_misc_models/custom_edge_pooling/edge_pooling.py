import torch
import numpy as np
from heapq import heappop, heapify
from torch import nn
import open3d as o3d

visualize = False
ignore_visualization = True

class MeshPool(nn.Module):

    def __init__(self, target=0, multi_thread=False):
        super(MeshPool, self).__init__()

    def __call__(self, x, edge_index, target_edge_count, batch, vertices, edges):
        return self.forward(x, edge_index, target_edge_count, batch, vertices, edges)

    @staticmethod
    def show_mesh(node_ids, node_ids_mask, index_offset, neighbours, vertices, edges, color_center, color_sides):
        return
        # if ignore_visualization:
        #     return
        print("Testing mesh in Open3D...")

        node_ids = node_ids[node_ids_mask]

        vertices = vertices.copy()
        # center in mean
        vertices[:, 0] = vertices[:, 0] - vertices[:, 0].min() - (vertices[:, 0].max() / 2)
        vertices[:, 1] = vertices[:, 1] - vertices[:, 1].min() - (vertices[:, 1].max() / 2)
        vertices[:, 2] = vertices[:, 2] - vertices[:, 2].min() - (vertices[:, 2].max() / 2)

        # old_node_id_to_maskes_id = {}
        # offset = 0
        # for i, masked in enumerate(node_ids_mask):
        #     if masked:
        #         old_node_id_to_maskes_id[i] = i - offset
        #     else:
        #         offset += 1

        triangles = []
        lines = []
        color_center_id = None
        color_sides_ids = []
        mesh_center_vertex_ids = []
        vertices_mask = np.ones(vertices.shape[0], dtype=np.bool)
        i = 0
        for node_id in node_ids:
            node_id = int(node_id)
            nbors = neighbours[node_id]

            lines.append(edges[node_id - index_offset].tolist())
            if color_center and node_id == color_center:
                color_center_id = i
                mesh_center_vertex_ids = edges[node_id - index_offset].tolist()
            if color_sides and node_id in color_sides:
                color_sides_ids.append(i)
            i += 1

            sides = []
            used = []
            for id_a in range(len(nbors)):
                if nbors[id_a] in used:
                    continue
                for id_b in range(id_a + 1, len(nbors)):
                    nb_a = nbors[id_a]
                    nb_b = nbors[id_b]
                    if nb_b in neighbours[nb_a]:
                        sides.append([nb_a, nb_b])
                        used.append(nb_a)
                        used.append(nb_b)
                        break

            # sides = []
            # side1_collapse_to_node = None
            # used = []
            # for id_a in range(len(nbors)):
            #     if nbors[id_a] in used:
            #         continue
            #     for id_b in range(id_a + 1, len(nbors)):
            #         nb_a = nbors[id_a]
            #         nb_b = nbors[id_b]
            #         if nb_b in neighbours[nb_a]:
            #             if side1_collapse_to_node:
            #                 # check if nb_a has common vertices with the collapse_to_first edge
            #                 # if true select it as the first item of the side, otherwise select nb_b
            #                 has_common_vertice = False
            #                 for vert in edges[nb_a - index_offset]:
            #                     if vert in edges[side1_collapse_to_node - index_offset]:
            #                         has_common_vertice = True
            #                         break
            #                 debug = edges[nb_b - index_offset]
            #                 debug2 = edges[side1_collapse_to_node - index_offset]
            #                 if has_common_vertice:
            #                     sides.append([nb_a, nb_b])
            #                 else:
            #                     sides.append([nb_b, nb_a])
            #             else:
            #                 sides.append([nb_a, nb_b])
            #                 side1_collapse_to_node = nb_a
            #             used.append(nb_a)
            #             used.append(nb_b)
            #             break

            for side in sides:
                s1 = side[0]
                s2 = side[1]
                #if s1 > node_id and s2 > node_id:
                triangle = edges[node_id - index_offset].tolist()
                if len(triangle) != 2:
                    print("asd")
                debug1 = edges[node_id - index_offset].tolist()
                debug2 = edges[s1 - index_offset]
                for e_id in edges[s1 - index_offset]:
                    if e_id not in triangle:
                        triangle.append(e_id)
                if len(triangle) != 3:
                    print("asd")
                #edges[s2 - offset_index]
                #triangles.append(sorted(triangle))
                triangles.append(triangle)
                for ti in triangle:
                    vertices_mask[ti] = False

        # set unused vertices to zero position
        vertices[vertices_mask] = [0, 0, 0]

        #vertex_colors
        for t in triangles:
            if len(t) != 3:
                print("ja")

        mesh_vertices = vertices.copy()
        mesh_vertices[:, 0] = vertices[:, 0] + (vertices[:, 0].max() - vertices[:, 0].min()) * 1.1
        # mesh_vertices[:, 0] = (mesh_vertices[:, 0] * 0.98) - (mesh_vertices[:, 0].max() - mesh_vertices[:, 0].min() / 2) + ((mesh_vertices[:, 0] * 0.98).max() - (mesh_vertices[:, 0] * 0.98).min() / 2)
        # mesh_vertices[:, 1] = (mesh_vertices[:, 1] * 0.98) - (mesh_vertices[:, 1].max() - mesh_vertices[:, 1].min() / 2) + ((mesh_vertices[:, 1] * 0.98).max() - (mesh_vertices[:, 1] * 0.98).min() / 2)
        # mesh_vertices[:, 2] = (mesh_vertices[:, 2] * 0.98) - (mesh_vertices[:, 2].max() - mesh_vertices[:, 2].min() / 2) + ((mesh_vertices[:, 2] * 0.98).max() - (mesh_vertices[:, 2] * 0.98).min() / 2)
        # mesh_vertices[:, 1] = mesh_vertices * 0.98
        # mesh_vertices[:, 2] = mesh_vertices * 0.98
        #mesh_vertices = mesh_vertices * 0.96
        # mesh_vertices[:, 0] = mesh_vertices[:, 0] - mesh_vertices[:, 0].min() - (mesh_vertices[:, 0].max() / 2)
        # mesh_vertices[:, 1] = mesh_vertices[:, 1] - mesh_vertices[:, 1].min() - (mesh_vertices[:, 1].max() / 2)
        # mesh_vertices[:, 2] = mesh_vertices[:, 2] - mesh_vertices[:, 2].min() - (mesh_vertices[:, 2].max() / 2)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh_vertices),
            o3d.utility.Vector3iVector(triangles),
        )
        #mesh.paint_uniform_color([0.8, 0.8, 0.8])
        #mesh_colors = np.asarray(mesh.vertex_colors)
        #mesh_colors[mesh_center_vertex_ids] = [0.8, 0, 0]
        #mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
        #mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.compute_vertex_normals()
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
        pc.paint_uniform_color([0.6, 0.6, 0.6])  # [1, 0.706, 0]
        colors = np.asarray(pc.colors)
        for csi in color_sides_ids:
            for line in lines[csi]:
                colors[line] = [0, 0, 0]
        pc.colors = o3d.utility.Vector3dVector(colors)
        #lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        lineset = o3d.geometry.LineSet(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector2iVector(lines))

        #open3d.cpu.pybind.utility.Vector3dVector, lines: open3d.cpu.pybind.utility.Vector2iVector

        lineset.paint_uniform_color([0.6, 0.6, 0.6])
        lineset_colors = np.asarray(lineset.colors)
        if color_center_id:
            lineset_colors[color_center_id] = [1, 0, 0]
        for cs in color_sides_ids:
            lineset_colors[cs] = [0, 0, 0]
        lineset.colors = o3d.utility.Vector3dVector(lineset_colors)

        # color_center_id = None
        # color_sides_ids = []

        #o3d.visualization.draw_geometries([mesh, pc, lineset])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.line_width = 20.0
        opt.point_size = 10.0
        vis.add_geometry(mesh)
        vis.add_geometry(pc)
        vis.add_geometry(lineset)
        vis.run()
        vis.destroy_window()

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

        # for nid in range(x.size(0)):
        #     sides = []
        #     side1_collapse_to_node = None
        #     used = []
        #     for id_a in range(len(neighbours)):
        #         if neighbour_ids[id_a] in used:
        #             continue
        #         for id_b in range(id_a + 1, len(neighbour_ids)):
        #             nb_a = neighbour_ids[id_a]
        #             nb_b = neighbour_ids[id_b]
        #             if nb_b in neighbours[nb_a]:
        #                 if side1_collapse_to_node:
        #                     # check if nb_a has common vertices with the collapse_to_first edge
        #                     # if true select it as the first item of the side, otherwise select nb_b
        #                     has_common_vertice = False
        #                     for vert in edges[batch_id][nb_a - index_offset.item()]:
        #                         if vert in edges[batch_id][side1_collapse_to_node - index_offset.item()]:
        #                             has_common_vertice = True
        #                             break
        #                     debug = edges[batch_id][nb_b - index_offset.item()]
        #                     debug2 = edges[batch_id][side1_collapse_to_node - index_offset.item()]
        #                     if has_common_vertice:
        #                         sides.append([nb_a, nb_b])
        #                     else:
        #                         sides.append([nb_b, nb_a])
        #                 else:
        #                     sides.append([nb_a, nb_b])
        #                     side1_collapse_to_node = nb_a
        #                 used.append(nb_a)
        #                 used.append(nb_b)
        #                 break
        #         if len(sides) < 2:
        #             print("ja index", nide)

        for batch_id in range(batch.max().item() + 1):
            node_ids = (batch == batch_id).nonzero().view(-1)
            node_ids_mask = np.ones(node_ids.size(0), dtype=np.bool)
            index_offset = node_ids.min()
            num_nodes = node_ids.size(0)
            queue = self.build_queue(x, node_ids)

            while num_nodes > target_edge_count:
                if num_nodes - 3 < target_edge_count:
                    self.show_mesh(node_ids, node_ids_mask, index_offset.item(), neighbours, vertices[batch_id],
                                   edges[batch_id],
                                   node_id, neighbour_ids)
                if len(queue) == 0:
                    print("queue is empty?!")
                    self.show_mesh(node_ids, node_ids_mask, index_offset.item(), neighbours, vertices[batch_id],
                                   edges[batch_id],
                                   node_id, neighbour_ids)
                    break

                value, node_id = heappop(queue)
                node_id = int(node_id)
                if value == None:
                    break

                if collapse_mask[node_id]:
                    collapse_mask[node_id] = False
                    neighbour_ids = neighbours[node_id]

                    # if neighbour_ids[1] in neighbours[neighbour_ids[0]]:
                    #     sides.append([neighbour_ids[0], neighbour_ids[1]])
                    #     if neighbour_ids[2] in neighbours[neighbour_ids[3]]:
                    #         sides.append([neighbour_ids[2], neighbour_ids[3]])
                    # elif neighbour_ids[2] in neighbours[neighbour_ids[0]]:
                    #     sides.append([neighbour_ids[0], neighbour_ids[2]])
                    #     if neighbour_ids[1] in neighbours[neighbour_ids[3]]:
                    #         sides.append([neighbour_ids[1], neighbour_ids[3]])
                    # elif neighbour_ids[3] in neighbours[neighbour_ids[0]]:
                    #     sides.append([neighbour_ids[0], neighbour_ids[3]])
                    #     if neighbour_ids[1] in neighbours[neighbour_ids[2]]:
                    #         sides.append([neighbour_ids[1], neighbour_ids[2]])

                    sides = []
                    side1_collapse_to_node = None
                    used = []
                    for id_a in range(len(neighbour_ids)):
                        if neighbour_ids[id_a] in used:
                            continue
                        for id_b in range(id_a + 1, len(neighbour_ids)):
                            nb_a = neighbour_ids[id_a]
                            nb_b = neighbour_ids[id_b]
                            if nb_b in neighbours[nb_a]:
                                if side1_collapse_to_node:
                                    # check if nb_a has common vertices with the collapse_to_first edge
                                    # if true select it as the first item of the side, otherwise select nb_b
                                    has_common_vertice = False
                                    for vert in edges[batch_id][nb_a - index_offset.item()]:
                                        if vert in edges[batch_id][side1_collapse_to_node - index_offset.item()]:
                                            has_common_vertice = True
                                            break
                                    debug = edges[batch_id][nb_b - index_offset.item()]
                                    debug2 = edges[batch_id][side1_collapse_to_node - index_offset.item()]
                                    if has_common_vertice:
                                        sides.append([nb_a, nb_b])
                                    else:
                                        sides.append([nb_b, nb_a])
                                else:
                                    sides.append([nb_a, nb_b])
                                    side1_collapse_to_node = nb_a
                                used.append(nb_a)
                                used.append(nb_b)
                                break

                    if len(sides) == 0:
                        print("broken?! hole?!")  # todo check this
                        if not ignore_visualization:
                            self.show_mesh(node_ids, node_ids_mask, index_offset.item(), neighbours, vertices[batch_id], edges[batch_id],
                                           node_id, neighbour_ids)
                        continue

                    if len(sides) == 1:
                        if not ignore_visualization:
                            self.show_mesh(node_ids, node_ids_mask, index_offset.item(), neighbours, vertices[batch_id],
                                           edges[batch_id],
                                           node_id, sides[0])
                        continue

                    if visualize:
                        flattened_sides = sides[0] + sides[1] if len(sides) > 1 else side[0]
                        self.show_mesh(node_ids, node_ids_mask, index_offset.item(), neighbours, vertices[batch_id], edges[batch_id], node_id, flattened_sides)

                    # temp ignore for now, solve later
                    ignore = False
                    if sides[0][0] in sides[1] or sides[0][1] in sides[1]:
                        #print("ignore0")
                        continue
                    for side in sides:
                        debug_s1_nbors_foo = neighbours[side[0]]
                        debug_s2_nbors_foo = neighbours[side[1]]
                        s1_outer_neighbours = [n for n in neighbours[side[0]] if n != node_id and n != side[1]]
                        s2_outer_neighbours = [n for n in neighbours[side[1]] if n != node_id and n != side[0]]
                        for on in s1_outer_neighbours:
                            if on in s2_outer_neighbours:
                                ignore = True
                    for side in [[sides[0][0], sides[1][0]], [sides[0][1], sides[1][1]]]:
                        s1_outer_neighbours = [n for n in neighbours[side[0]] if n != node_id and n != side[1]]
                        s2_outer_neighbours = [n for n in neighbours[side[1]] if n != node_id and n != side[0]]
                        for on in s1_outer_neighbours:
                            if on in s2_outer_neighbours:
                                ignore = True
                    if ignore:
                        #print("ignore1")
                        continue
                    # end temp ignore

                    node_ids_mask[node_id - index_offset.item()] = False  # do this after show mesh
                    x_mask[node_id] = False  # delete current node
                    num_nodes -= 1  # for current edge

                    for side in sides:
                        num_nodes -= 1  # for each side that is collapsed
                        s1 = side[0]
                        s2 = side[1]
                        # merge s2 into s1
                        #print("collapse", node_id, "and merge", s2, "into", s1)

                        # check and clean triplets
                        # triplet = True
                        # while triplet:
                        #     triplet = False
                        #     for side_nbor in neighbours[s1]:
                        #         if side_nbor == node_id or side_nbor == s2:
                        #             continue
                        #         if side_nbor in neighbours[s2]:
                        #             # num_common_neigbous_with_node = 0
                        #             # for c_nb in neighbours[side_nbor]:
                        #             #     if c_nb == s1 or c_nb == s2:
                        #             #         continue
                        #             #     if c_nb in neighbours[node_id]:
                        #             #         num_common_neigbous_with_node += 1
                        #             # if num_common_neigbous_with_node < 2:
                        #             #     continue
                        #             debug_nbs_s1 = neighbours[s1]
                        #             debug_nbs_s2 = neighbours[s2]
                        #             # it's a triples, remove s1, s2 and the common edge neighbour and
                        #             # use the outer triple edges as s1 and s2
                        #
                        #             # node_ids_mask[node_id - index_offset.item()] = True  # do this after show mesh
                        #             # flattened_sides = sides[0] + sides[1] if len(sides) > 1 else side[0]
                        #             # flattened_sides += [side_nbor]
                        #             # self.show_mesh(node_ids, node_ids_mask, index_offset.item(), neighbours,
                        #             #                vertices[batch_id], edges[batch_id], node_id, flattened_sides)
                        #             # node_ids_mask[node_id - index_offset.item()] = False  # do this after show mesh
                        #
                        #             # find out new s1 and s2
                        #             new_s = []
                        #             for s in [s1, s2]:
                        #                 for ns in neighbours[s]:
                        #                     if ns in neighbours[side_nbor] and ns != s1 and ns != s2:
                        #                         new_s.append(ns)
                        #             old_s1 = s1
                        #             old_s2 = s2
                        #             if len(new_s) < 2:
                        #                 self.show_mesh(node_ids, node_ids_mask, index_offset.item(), neighbours,
                        #                                vertices[batch_id],
                        #                                edges[batch_id],
                        #                                node_id, [old_s1, old_s2, s1, s2])
                        #                 print("asdsad")
                        #             s1 = new_s[0]
                        #             s2 = new_s[1]
                        #             debug_new_s1 = s1
                        #             debug_new_s2 = s2
                        #             neighbours[s1] = [n for n in neighbours[s1] if n != old_s1 and n != side_nbor]
                        #             neighbours[s2] = [n for n in neighbours[s2] if n != old_s2 and n != side_nbor]
                        #             debug_neighbours_s1_new = neighbours[s1]
                        #             debug_neighbours_s2_new = neighbours[s2]
                        #             triplet = True
                        #
                        #             x_mask[old_s1] = False  # delete node s1
                        #             node_ids_mask[old_s1 - index_offset.item()] = False  # hide from visualization
                        #             collapse_mask[old_s1] = False
                        #             x_mask[old_s2] = False  # delete node old_s2
                        #             node_ids_mask[old_s2 - index_offset.item()] = False  # hide from visualization
                        #             collapse_mask[old_s2] = False
                        #             x_mask[side_nbor] = False  # delete node side_nbor
                        #             node_ids_mask[side_nbor - index_offset.item()] = False  # hide from visualization
                        #             collapse_mask[side_nbor] = False
                        #             num_nodes -= 3
                        #
                        #             break

                        # TODO: check if after multiple iterations it can be possible that s1 and s2 are the same!?

                        debug_node_nbs = neighbours[node_id]
                        debug_s1_nbs = neighbours[s1]
                        debug_s2_nbs = neighbours[s2]

                        collapse_mask[s1] = False  # are not available for collapsing in the iteration
                        collapse_mask[s2] = False  # are not available for collapsing in the iteration

                        # calculate new neighbours
                        new_neighbour_ids = []
                        for nb in neighbours[s1]:
                            # ignore node_id (collapsed) and s2 as s1 and s2 are beeing merged right now
                            if nb == node_id or nb == s2:
                                continue
                            new_neighbour_ids.append(nb)
                        for nb in neighbours[s2]:
                            # ignore node_id (collapsed) and s1 as s1 and s2 are beeing merged right now
                            if nb == node_id or nb == s1:
                                continue
                            new_neighbour_ids.append(nb)
                            # set reference of neighbour's neighbours form s2 to s1
                            debug_nb_neighbours = neighbours[nb]
                            neighbours[nb] = [n for n in neighbours[nb] if n != s2]  # remove ref of s2
                            if s1 not in neighbours[nb]: # this should always be the caase...
                                neighbours[nb].append(s1)  # add ref of s1
                            else:
                                node_ids_mask[node_id - index_offset.item()] = True  # do this after show mesh
                                self.show_mesh(node_ids, node_ids_mask, index_offset.item(), neighbours,
                                               vertices[batch_id],
                                               edges[batch_id], node_id, [s1, s2])
                            debug_nb_neighbours_2 = neighbours[nb]

                            # consider also updating the vertice position by taking the center position

                        if len(set(new_neighbour_ids)) != len(new_neighbour_ids):
                            print("ladida")

                        x[s1] = (x[node_id] + x[s1] + x[s2] / 3)  # update node s1
                        x_mask[s2] = False  # delete node s2
                        node_ids_mask[s2 - index_offset.item()] = False  # hide from visualization
                        neighbours[s1] = new_neighbour_ids  # set new neighbours


                        # set new vertice position (mean of merged)
                        # find different vertice_ids
                        # vertice_id_s1 = [vid for vid in edges[batch_id][s1 - index_offset] if vid not in edges[batch_id][s2 - index_offset]][0]
                        # vertice_id_s2 = [vid for vid in edges[batch_id][s2 - index_offset] if vid not in edges[batch_id][s1 - index_offset]][0]
                        # vertices[batch_id][vertice_id_s1 - index_offset] = vertices[batch_id][vertice_id_s1 - index_offset] - ((vertices[batch_id][vertice_id_s1 - index_offset] - vertices[batch_id][vertice_id_s2 - index_offset]) / 2)

                    # from_vertice_id = [ve_id for ve_id in edges[batch_id][node_id - index_offset.item()] if ve_id in edges[batch_id][s2 - index_offset.item()]][0]
                    # to_vertice_id = [ve_id for ve_id in edges[batch_id][node_id - index_offset.item()] if ve_id != from_vertice_id]
                    # if len(to_vertice_id) == 0:
                    #     # should not happen?!
                    #     self.show_mesh(node_ids, node_ids_mask, index_offset.item(), neighbours, vertices[batch_id],
                    #                    edges[batch_id], node_id, sides[0])

                    # to_vertice_id = to_vertice_id[0]

                    def update_vertice_id(node):
                        node_neighbours = neighbours[node]
                        for n in node_neighbours:
                            for i in range(2):
                                vertice_id = edges[batch_id][n - index_offset.item()][i]
                                if vertice_id == from_vertice_id:
                                    edges[batch_id][n - index_offset.item()][i] = to_vertice_id
                                    update_vertice_id(n)
                    # update_vertice_id(s2)

                    if len(sides) < 2:
                        print("LESS THEN TWO")
                        self.show_mesh(node_ids, node_ids_mask, index_offset.item(), neighbours, vertices[batch_id],
                                       edges[batch_id], node_id, sides[0])
                        for nib in neighbour_ids:
                            if nib not in used:
                                # not collapsed, probably an artifact, remove edge to node_id
                                neighbours[nib] = [n for n in neighbours[nib] if n != node_id]
                                print("ladad")

                    if visualize:
                        flattened_sides = [sides[0][0], sides[1][0]] if len(sides) > 1 else side[0][0]
                        self.show_mesh(node_ids, node_ids_mask, index_offset.item(), neighbours, vertices[batch_id],
                                       edges[batch_id], None, flattened_sides)

        masked_ids = torch.masked_select(torch.range(0, len(x_mask) - 1), torch.tensor(x_mask))
        #x = torch.masked_select(x.T, torch.tensor(x_mask))
        #x = x.T[masked_ids]
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

        print(len(new_edge_index[0]))
        return x, torch.tensor(new_edge_index, dtype=torch.long).to(batch.device), batch

    def build_queue(self, features, batch_node_ids):
        #batch_node_ids = (batch == batch_id).nonzero().view(-1)
        squared_magnitude = torch.sum(features[batch_node_ids] * features[batch_node_ids], 1)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        #batch_node_ids = torch.arange(batch_node_ids, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, batch_node_ids.unsqueeze(-1)), dim=-1).tolist()
        heapify(heap)
        return heap