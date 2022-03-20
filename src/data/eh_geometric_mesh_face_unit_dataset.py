from __future__ import print_function
import os
import os.path
import torch
import csv
import open3d as o3d
from torch_geometric.data import Data, Dataset
from src.data.mesh_cnn_edge_feature_extractor import fill_mesh
from torch_geometric.transforms import NormalizeScale
from src.data.dataset_dir_reader import read_csv_dataset_directory
import numpy as np


class EHMeshDatasetGeometricFaceUnit(Dataset):
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

        # BUG previously: => during training config.test_transform and during test the test transform was used!
        # transform = config.test_transform if split == 'train' and config.test_transform else config.transform
        # FIXED
        transform = None
        if split == 'train' and config.transform:
            transform = config.transform
        elif split == 'test' and config.test_transform:
            transform = config.test_transform

        super(EHMeshDatasetGeometricFaceUnit, self).__init__(root, transform, None)

    @staticmethod
    def is_stl_extensions(sample_dir):
        for root, _, fnames in sorted(os.walk(sample_dir)):
            for fname in sorted(fnames):
                if fname.endswith('.stl'):
                    return True
                if fname.endswith('.obj'):
                    return False
        raise Exception("Neither STL or OBJ found")

    @staticmethod
    def cache_obj(sample_dir, cache_path):
        os.makedirs(os.path.join(sample_dir, cache_path), exist_ok=True)
        for root, _, fnames in sorted(os.walk(sample_dir)):
            for fname in sorted(fnames):
                target_obj_fname = fname.replace('.stl', '.obj')
                target_obj_path = os.path.join(root, cache_path, target_obj_fname)
                if fname.endswith('.stl') and not os.path.isfile(target_obj_path):
                    mesh = o3d.io.read_triangle_mesh(os.path.join(root, fname))
                    mesh = mesh.merge_close_vertices(0.0000001)  # necessary, otherwise triangles will not be connected
                    o3d.io.write_triangle_mesh(target_obj_path, mesh)

    def set_transform(self, transform):
        self.transform = transform

    def len(self):
        return len(self.models)

    def get(self, idx):
        model = self.models[idx]
        ring_edges, edge_features, vertices, edges, _edge_normals = fill_mesh(model['datapath'], self.config)
        edge_features = edge_features.T

        faces = []
        face_vertices = []
        edge_to_face_map = {}
        face_id_counter = 0

        # calculate edge_index list
        from_v_list = []
        to_v_list = []

        foo = len(ring_edges)
        for edge_from_id in range(len(ring_edges)):
            ring = ring_edges[edge_from_id]
            for side_a, side_b in [[ring[0], ring[1]], [ring[2], ring[3]]]:
                if side_a == -1 or side_b == -1:
                    continue
                if side_a < edge_from_id or side_b < edge_from_id:
                    continue
                edge_sides = [edge_from_id, side_a, side_b]
                vertices_a = torch.tensor([vertices[edges[edge_from_id][0]], vertices[edges[side_a][0]], vertices[edges[side_b][0]]])
                vertices_b = torch.tensor([vertices[edges[edge_from_id][1]], vertices[edges[side_a][1]], vertices[edges[side_b][1]]])
                lengths = torch.sum((vertices_a - vertices_b) * (vertices_a - vertices_b), axis=1)
                shorted_idx = torch.argmin(lengths)

                faces.append(np.concatenate((
                    edge_features[edge_sides[shorted_idx]],
                    edge_features[edge_sides[(shorted_idx + 1) % 3]],
                    edge_features[edge_sides[(shorted_idx + 2) % 3]],
                )))

                vertice_set = set()
                cur_face_vertices = []
                for s_id in edge_sides:
                    if s_id in edge_to_face_map:
                        from_v_list.append(face_id_counter)
                        to_v_list.append(edge_to_face_map[s_id])
                        from_v_list.append(edge_to_face_map[s_id])
                        to_v_list.append(face_id_counter)
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

        # get vertex normals
        x = torch.tensor(faces, dtype=torch.float32)

        if self.config.mesh_use_xyz:
            face_middle_xyz = 1/3 * torch.sum(torch.tensor(face_vertices, dtype=torch.float32), axis=1)
            pos = face_middle_xyz

            data_tmp = Data(pos=pos)
            if self.config.normalize:
                data_tmp = NormalizeScale()(data_tmp)
            pos = data_tmp.pos

        data = Data(x=x, pos=pos, edge_index=edge_index, y=torch.tensor([model['cls_idx']]).type(torch.long),
                    vertices=vertices,
                    edges=edges)

        return data