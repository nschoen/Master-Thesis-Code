from __future__ import print_function
import os
import os.path
import torch
import csv
import open3d as o3d
from torch_geometric.data import Data, Dataset
from src.data.mesh_cnn_edge_feature_extractor import fill_mesh
from torch_geometric.transforms import NormalizeScale
import torch.utils.data as data
from src.data.dataset_dir_reader import read_csv_dataset_directory
import numpy as np


class EHMeshDatasetGeometricFaceUnitEdgeAttr(Dataset):
    def __init__(self,
                 root,
                 config,
                 split='train',
                 cross_validation_set=False):
        self.root = root
        self.config = config
        self.transform = config.transform

        self.mesh_dir = config.mesh_dir
        if isinstance(self.mesh_dir, str):
            self.mesh_dir = [self.mesh_dir]

        for i in range(len(self.mesh_dir)):
            mesh_dir = os.path.join(root, self.mesh_dir[i])
            if self.is_stl_extensions(mesh_dir):
                self.cache_obj(mesh_dir, "obj-cached")
                mesh_dir = os.path.join(mesh_dir, "obj-cached")
            self.mesh_dir[i] = mesh_dir

        self.models, self.classid_map, self.classes = read_csv_dataset_directory(root,
                                                                                 self.config,
                                                                                 split,
                                                                                 cross_validation_set,
                                                                                 self.mesh_dir)

        transform = config.test_transform if split == 'train' and config.test_transform else config.transform
        super(EHMeshDatasetGeometricFaceUnitEdgeAttr, self).__init__(root, transform, None)

    def len(self):
        return len(self.models)

    def get(self, idx):
        model = self.models[idx]
        ring_edges, edge_features, vertices, edges = fill_mesh(model['datapath'], self.config)
        edge_features = edge_features.T

        face_vertices = []
        edge_to_face_map = {}
        face_id_counter = 0

        # calculate edge_index list
        from_v_list = []
        to_v_list = []
        edge_attr = []

        for edge_from_id in range(len(ring_edges)):
            ring = ring_edges[edge_from_id]
            for side_a, side_b in [[ring[0], ring[1]], [ring[2], ring[3]]]:
                if side_a == -1 or side_b == -1:
                    continue
                if side_a < edge_from_id or side_b < edge_from_id:
                    continue
                edge_sides = [edge_from_id, side_a, side_b]
                vertice_set = set()
                cur_face_vertices = []
                for s_id in edge_sides:
                    if s_id in edge_to_face_map:
                        from_v_list.append(face_id_counter)
                        to_v_list.append(edge_to_face_map[s_id])
                        edge_attr.append(edge_features[s_id])
                        from_v_list.append(edge_to_face_map[s_id])
                        to_v_list.append(face_id_counter)
                        edge_attr.append(edge_features[s_id])
                    else:
                        edge_to_face_map[s_id] = face_id_counter
                    if edges[s_id][0] not in vertice_set:
                        vertice_set.add(edges[s_id][0])
                        cur_face_vertices.append(vertices[edges[s_id][0]])
                    if edges[s_id][1] not in vertice_set:
                        vertice_set.add(edges[s_id][1])
                        cur_face_vertices.append(vertices[edges[s_id][1]])
                face_vertices.append(cur_face_vertices)

                # increment face id counter
                face_id_counter += 1

        edge_index = torch.tensor([from_v_list, to_v_list], dtype=torch.long)

        if self.config.mesh_use_xyz:
            face_middle_xyz = 1/3 * torch.sum(torch.tensor(face_vertices, dtype=torch.float32), axis=1)
            pos = face_middle_xyz

            data_tmp = Data(pos=pos)
            if self.config.normalize:
                data_tmp = NormalizeScale()(data_tmp)
            pos = data_tmp.pos

        data = Data(pos=pos,
                    edge_index=edge_index,
                    edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                    y=torch.tensor([model['cls_idx']]).type(torch.long),
                    vertices=vertices,
                    edges=edges)

        return data