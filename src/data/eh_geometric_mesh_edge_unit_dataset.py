from __future__ import print_function
import os
import os.path
import torch
import csv
import open3d as o3d
from torch_geometric.data import Data, Dataset
from src.data.mesh_cnn_edge_feature_extractor import fill_mesh
from torch_geometric.transforms import NormalizeScale
import numpy as np
import random


class EHMeshDatasetGeometricEdgeUnit(Dataset):
    def __init__(self,
                 root,
                 config,
                 split='train',
                 cross_validation_set=False,
                 load_dir=False):
        self.root = root
        self.config = config
        self.models = []
        self.classid_map = {}
        self.transform = config.transform
        self.last_model = None
        self.last_index = None
        self.self_loop = True if not hasattr(self.config, 'self_loop') else self.config.self_loop
        self.use_broken_self_loop = False if not hasattr(self.config, 'use_broken_self_loop') else self.config.use_broken_self_loop

        if self.self_loop and not self.use_broken_self_loop:
            print("using self loop")
        elif self.use_broken_self_loop:
            print("using BROKEN self loop")

        self.mesh_dir = config.mesh_dir
        if isinstance(self.mesh_dir, str):
            self.mesh_dir = [self.mesh_dir]

        for i in range(len(self.mesh_dir)):
            mesh_dir = os.path.join(root, self.mesh_dir[i])
            if self.is_stl_extensions(mesh_dir):
                self.cache_obj(mesh_dir, "obj-cached")
                mesh_dir = os.path.join(mesh_dir, "obj-cached")
            self.mesh_dir[i] = mesh_dir

        if not load_dir:
            with open(os.path.join(root, config.models_csv), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                for row in reader:
                    if cross_validation_set and (
                        (split == 'train' and int(row[4]) == cross_validation_set) or (split == 'test' and int(row[4]) != cross_validation_set)
                    ) or not cross_validation_set and row[3] != split:
                        continue

                    self.classid_map[int(row[2])] = row[1]
                    if config.filter_classes and int(row[2]) not in config.filter_classes:
                        continue

                    filename = row[0]
                    basename = filename.split('.')[0]
                    filename = f"{basename}.obj"

                    if config.filename_filter and basename not in config.filename_filter:
                        continue

                    if config.mesh_dir_select_random is not None and config.mesh_dir_select_random > 0:
                        self.mesh_dir_select_random = True
                        for item in range(config.mesh_dir_select_random):
                            self.models.append({
                                'cls': row[1],
                                'cls_idx': int(row[2]),
                                'datapath': '',
                                'filename': filename,
                                #'select_random_from_meshdirs': True,
                            })
                    else:
                        self.mesh_dir_select_random = False
                        for mesh_dir in self.mesh_dir:
                            datapath = os.path.join(mesh_dir, filename)

                            self.models.append({
                                'cls': row[1],
                                'cls_idx': int(row[2]),
                                'datapath': datapath,
                                'filename': filename,
                            })

                # create sorted list of classes
                class_ids = list(self.classid_map.keys())
                class_ids.sort()
                self.classes = [self.classid_map[idx] for idx in class_ids]
        else:
            # just read provided folder for inference
            file_classes = {}
            self.mesh_dir_select_random = False
            with open(os.path.join(root, self.config.models_csv), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                for row in reader:
                    self.classid_map[int(row[2])] = row[1]
                    basename = row[0].split('.')[0]
                    file_classes[basename] = {
                        'cls': row[1],
                        'cls_idx': int(row[2]),
                    }
            for _root, dirs, files in os.walk(root, topdown=False):
                for file in files:
                    if not file.endswith('stl') and not file.endswith('obj'):
                        continue
                    basename = file.split('.')[0]
                    self.models.append({
                        'cls': file_classes[basename]['cls'] if basename in file_classes.keys() else '',
                        'cls_idx': file_classes[basename]['cls_idx'] if basename in file_classes.keys() else -1,
                        'datapath': os.path.join(root, file),
                        'filename': file,
                    })

        # BUG previously: => during training config.test_transform and during test the test transform was used!
        # transform = config.test_transform if split == 'train' and config.test_transform else config.transform
        # FIXED
        transform = None
        if split == 'train' and config.transform:
            transform = config.transform
        elif split == 'test' and config.test_transform:
            transform = config.test_transform

        super(EHMeshDatasetGeometricEdgeUnit, self).__init__(root, transform, None)

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
        self.last_model = model
        self.last_index = idx
        model_datapath = model['datapath']
        if self.mesh_dir_select_random:
            random_mesh_dir = random.choice(self.mesh_dir)
            model_datapath = os.path.join(random_mesh_dir, model['filename'])
        ring_edges, edge_features, vertices, edges, edge_normals = fill_mesh(model_datapath, self.config)


        # calculate edge_index list
        from_v_list = []
        to_v_list = []
        edge_attr = None
        edge_set = set()
        if self.config.mesh_edge_normals:
            edge_attr = []

        for edge_from_id in range(len(ring_edges)):
            # add self-loop
            if self.self_loop and not self.use_broken_self_loop:
                from_v_list.append(edge_from_id)
                to_v_list.append(edge_from_id)
                if self.config.mesh_edge_normals:
                    edge_attr.append(0.0)

            # add edges to ring neighbours
            ring = ring_edges[edge_from_id]
            for edge_to_id in ring:
                if edge_to_id == -1:
                    continue
                if edge_from_id == edge_to_id:
                    assert False, "Not allowed"
                edge_set_id = f"{edge_from_id}-{edge_to_id}"
                if edge_set_id in edge_set:
                    #assert False, "double edge?"
                    #print("double edge...")
                    continue
                edge_set.add(edge_set_id)
                from_v_list.append(edge_from_id)
                to_v_list.append(edge_to_id)
                if self.config.mesh_edge_normals:
                    normal_angle = np.arccos(np.clip(np.dot(edge_normals[edge_from_id], edge_normals[edge_to_id]), -1.0, 1.0))
                    edge_attr.append(normal_angle)

                if self.use_broken_self_loop:
                    from_v_list.append(edge_from_id)
                    to_v_list.append(edge_from_id)
                    if self.config.mesh_edge_normals:
                        edge_attr.append(0.0)

        edge_index = torch.tensor([from_v_list, to_v_list], dtype=torch.long)
        if self.config.mesh_edge_normals:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        # get vertex normals
        x = torch.tensor(edge_features.T, dtype=torch.float32)

        if self.config.mesh_use_xyz:
            #edge_middle_xyz = vertices[edges[:, 0]] - (vertices[edges[:, 0]] - vertices[edges[:, 1]]) / 2
            edge_middle_xyz = (vertices[edges[:, 0]] + vertices[edges[:, 1]]) / 2
            pos = torch.tensor(edge_middle_xyz, dtype=torch.float32)

            data_tmp = Data(pos=pos)
            if self.config.normalize:
                data_tmp = NormalizeScale()(data_tmp)
            pos = data_tmp.pos

        normal = None
        if self.config.mesh_edge_normals:
            normal = torch.tensor(edge_normals.T, dtype=torch.float32).T

        data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([model['cls_idx']]).type(torch.long),
                    vertices=vertices, neighbour_ids=ring_edges, edges=edges, normal=normal)
                    #keys=['x', 'pos', 'edge_index', 'edge_attr', 'y'])

        return data