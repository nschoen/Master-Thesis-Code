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
from src.data.eh_geometric_mesh_edge_unit_dataset import EHMeshDatasetGeometricEdgeUnit
from src.data.eh_pointcloud_dataset import EHDataset


class EHPointcloudEdgeUnitDataset(Dataset):
    def __init__(self,
                 root,
                 config,
                 split='train',
                 cross_validation_set=False):

        config.mesh_use_xyz = True
        self.config = config
        assert self.config.n_points
        self.geometric_ds = EHMeshDatasetGeometricEdgeUnit(root,
                                                           config,
                                                           split=split,
                                                           cross_validation_set=cross_validation_set)

        self.models = self.geometric_ds.models
        self.classid_map = self.geometric_ds.classid_map
        self.classes = self.geometric_ds.classes

        super(EHPointcloudEdgeUnitDataset, self).__init__(root, None, None)

    def set_transform(self, transform):
        self.geometric_ds.set_transform(transform)

    def len(self):
        return self.geometric_ds.__len__()

    def get(self, idx):
        model = self.geometric_ds.models[idx]
        data = self.geometric_ds.__getitem__(idx)

        point_set = data.pos.numpy().astype(np.float32)
        normals = data.normal.numpy().astype(np.float32)
        point_set, last_transformations, normals = EHDataset.augment_point_set(point_set,
                                                                      normalize=self.config.normalize,
                                                                      normalize_type=self.config.normalize_type,
                                                                      normalize_scale_factor=self.config.normalize_scale_factor,
                                                                      rotate_axes=self.config.rotate_axes,
                                                                      move_along_axis=self.config.move_along_axis,
                                                                      data_augmentation_jitter=self.config.data_augmentation_jitter,
                                                                      normals=normals)

        self.last_transformations = last_transformations
        point_set = torch.from_numpy(point_set)
        normals = torch.from_numpy(normals)

        points = torch.cat((point_set, normals, data.x), axis=-1)

        # if self.config.wrap_cls:
        return points, np.array([model['cls_idx']]).astype(np.long)