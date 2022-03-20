from __future__ import print_function
import os
import os.path
import torch
import numpy as np
import csv
import open3d as o3d
from torch_geometric.data import Data, Dataset
from src.data.feature_extrator_edge import extract_edge_features
import multiprocessing
from src.data.eh_pointcloud_dataset import EHDataset

def calculate_mesh_features(simplified_mesh_file, mesh_feature_file):
    mesh = o3d.io.read_triangle_mesh(simplified_mesh_file)

    # vertices_set = np.asarray(mesh.vertices).astype(np.float32)
    # x = torch.tensor(vertices_set).type(torch.float32)

    # calculate edge_features
    edge_features = torch.tensor(extract_edge_features(mesh).T)

    # calculate vertex normals
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.normalize_normals()
    normal_features = torch.tensor(np.asarray(mesh.vertex_normals)).type(torch.float32)

    torch.save({
        'edge_features': edge_features,
        'vertex_normals': normal_features,
    }, mesh_feature_file)

    return True


class EHMeshDatasetGeometric(Dataset):
    def __init__(self,
                 root,
                 config,
                 split='train',
                 cross_validation_set=False):
        self.root = root
        self.config = config
        self.mesh_dir = os.path.join(self.root, 'data-obj')

        if config.max_number_triangles:
            self.original_mesh_dir = self.mesh_dir
            self.mesh_dir = f"{self.mesh_dir}-{config.max_number_triangles}-cached"
        elif config.explicit_mesh_dir:
            self.mesh_dir = os.path.join(self.root, config.explicit_mesh_dir)

        self.models = []
        self.classid_map = {}
        self.transform = config.transform

        if not os.path.isdir(self.mesh_dir):
            os.makedirs(self.mesh_dir, exist_ok=True)

        pool = multiprocessing.Pool(64)
        jobs = []

        with open(os.path.join(root, config.dataset_classes_csv), newline='') as csvfile:
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
                filename = f"{basename}.{config.mesh_extension}"

                if config.filename_filter and basename not in config.filename_filter:
                    continue

                datapath = os.path.join(self.mesh_dir, filename)

                simplified_mesh_file = os.path.join(self.mesh_dir, f"{basename}.{config.mesh_extension}")
                if config.max_number_triangles:
                    # check if mesh with target number of triangles exist, if not, create it
                    if not os.path.isfile(simplified_mesh_file):
                        original_datapath = os.path.join(self.original_mesh_dir, filename)
                        mesh = o3d.io.read_triangle_mesh(original_datapath)
                        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=config.max_number_triangles)
                        o3d.io.write_triangle_mesh(simplified_mesh_file, mesh)

                mesh_feature_file = os.path.join(self.mesh_dir, f"{basename}_feature_cache.pyt")

                #if os.path.isfile(mesh_feature_file):
                #    os.remove(mesh_feature_file)

                if not os.path.isfile(mesh_feature_file):
                    print("simplified_mesh_file", simplified_mesh_file)
                    job = pool.apply_async(calculate_mesh_features, (simplified_mesh_file, mesh_feature_file, ))
                    jobs.append(job)

                self.models.append({
                    'cls': row[1],
                    'cls_idx': int(row[2]),
                    'datapath': datapath,
                    'filename': filename,
                })

            i = 0
            for j in jobs:
                print("start new job", i)
                j.get()
                i += 1

            pool.terminate()
            pool.close()

            # create sorted list of classes
            class_ids = list(self.classid_map.keys())
            class_ids.sort()
            self.classes = [self.classid_map[idx] for idx in class_ids]

        super(EHMeshDatasetGeometric, self).__init__(root, transform, None)

    def set_transform(self, transform):
        self.transform = transform

    def len(self):
        return len(self.models)

    def get(self, idx):
        model = self.models[idx]
        mesh = o3d.io.read_triangle_mesh(model['datapath'])
        mesh = mesh.merge_close_vertices(0.000000000001) # TODO, do this in preprocessing
        mesh_features = torch.load(model['datapath'].replace(f".{self.config.extension}", '_feature_cache.pyt'))

        # get vertices x
        vertices_set = np.asarray(mesh.vertices).astype(np.float32)

        # manipulate coordinates
        vertices_set, last_transformations = EHDataset.augment_point_set(vertices_set,
                                                                         normalize=self.config.normalize,
                                                                         normalize_type=self.config.normalize_type,
                                                                         normalize_scale_factor=self.config.normalize_scale_factor,
                                                                         rotate_axes=self.config.rotate_axes,
                                                                         move_along_axis=self.config.move_along_axis,
                                                                         data_augmentation_jitter=self.config.data_augmentation_jitter)

        # self.last_transformations = last_transformations
        x = torch.tensor(vertices_set).type(torch.float32)
        pos = x  # torch.tensor(vertices_set).type(torch.float32)

        # calculate edge_index list
        edge_features = None
        indices = []
        # if self.config.edge_relative_angle_and_ratio:
        #     edge_features = mesh_features['edge_features']
        #     # edge_features = torch.tensor(extract_edge_features(mesh).T)
        if self.config.edge_relative_angle_and_ratio:
            indices += [0, 1, 2, 3, 4]
        if self.config.edge_length:
            indices += [5]
        if self.config.edge_vertice_normals_cosine:
            indices += [6]
        if len(indices) > 0:
            edge_features = mesh_features['edge_features'][:, indices].type(torch.float32)

        mesh.compute_adjacency_list()
        adj_list = mesh.adjacency_list
        from_v_list = []
        to_v_list = []
        for from_v in range(len(adj_list)):
            for to_v in adj_list[from_v]:
                from_v_list.append(from_v)
                to_v_list.append(to_v)
        edge_index = torch.tensor([from_v_list, to_v_list], dtype=torch.long)  # .type(torch.long)

        # get vertex normals
        if self.config.vertex_normals:
            # feed augmented vertices set back to mesh and calculate normals, attach them to the xyz features in x
            normal_features = mesh_features['vertex_normals']
            x = torch.cat((x, normal_features), dim=1)

        if not self.config.vertex_xyz:
            if self.config.vertex_one_vector:
                x = x[:, :3] = torch.ones(x.size(0), 3)
            else:
                # remove xyz
                x = x[:, 3:]

        # IDEA: setting for adding default vector of ones

        data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_features, y=torch.tensor([model['cls_idx']]).type(torch.long))

        return data