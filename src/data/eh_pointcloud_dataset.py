from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from src.utils.pointcloud_utils import pointcloud_utils as pc_utils
import csv


class EHDataset(data.Dataset):
    def __init__(self,
                 root,
                 config,
                 split='train',
                 cross_validation_set=False,
                 load_dir=False):
        self.config = config
        self.sampling_method = self.config.point_sampling_method
        self.models = []
        self.classes = []
        self.classid_map = {}
        self.cache_dir = self.get_cache_dir(root)
        self.last_index = None
        self.last_model = None

        file_extension = None

        if not load_dir:
            # read models from csv file
            with open(os.path.join(root, self.config.models_csv), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                for row in reader:
                    # if model does not belong to current set (test/train or cross validation set), ignore it
                    if cross_validation_set and (
                        (split == 'train' and int(row[4]) == cross_validation_set) or (split == 'test' and int(row[4]) != cross_validation_set)
                    ) or not cross_validation_set and row[3] != split and split != 'all':
                        continue

                    # add entry to map class id to class name
                    self.classid_map[int(row[2])] = row[1]

                    # filter class if set
                    if config.filter_classes and int(row[2]) not in config.filter_classes:
                        continue

                    filename = row[0]
                    basename = filename.split('.')[0]

                    # filter by filename if set
                    if config.filename_filter and basename not in config.filename_filter:
                        continue

                    if not file_extension:
                        # detect mesh file extension
                        for ext in ['stl', 'off', 'obj']:
                            datapath = os.path.join(root, config.mesh_dir, f"{basename}.{ext}")
                            if os.path.isfile(datapath):
                                file_extension = ext
                                break

                    # set datapath to original mesh
                    datapath = os.path.join(root, config.mesh_dir, f"{basename}.{file_extension}")

                    self.models.append({
                        'cls': row[1],
                        'cls_idx': int(row[2]),
                        'datapath': datapath,
                        'filename': filename,
                    })
        else:
            # just read provided folder for inference
            file_classes = {}
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

        # create sorted list of classes
        class_ids = list(self.classid_map.keys())
        class_ids.sort()
        self.classes = [self.classid_map[idx] for idx in class_ids]

    def get_cache_dir(self, root):
        # create cache directory for the sampled point clouds
        if self.sampling_method == 'poisson':
            cache_dir = os.path.join(root, 'cache', f"pc-poisson-{self.config.n_points}")
        elif self.sampling_method == 'nodes_plus_poisson':
            suffix = f"-{self.config.mesh_dir}" if self.config.mesh_dir else ''
            cache_dir = os.path.join(root,
                                     'cache',
                                     f"pc-node-poisson-{self.config.n_points}-{self.config.n_min_surface_sampled}{suffix}")
        # elif self.sampling_method == 'face_center_plus_poisson':
        #     suffix = f"-{self.config.mesh_dir}" if self.config.mesh_dir else ''
        #     cache_dir = os.path.join(root,
        #                              'cache',
        #                              f"pc-face-center-plus-poisson-{self.config.n_points}-{self.config.n_min_surface_sampled}{suffix}")
        else:
            raise Exception("Unknown point sampling method provided")
        if cache_dir and not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def __getitem__(self, index):
        model = self.models[index]
        self.last_index = index
        self.last_model = model

        # create point cloud
        if self.sampling_method == 'poisson':
            points, normals = pc_utils.sample_poisson_from_mesh(model['datapath'],
                                                                n_points=self.config.n_points,
                                                                cache_dir=self.cache_dir)
        elif self.sampling_method == 'nodes_plus_poisson':
            points, normals = pc_utils.sample_nodes_and_poisson_from_mesh(model['datapath'],
                                                                          n_points_total=self.config.n_points,
                                                                          n_min_poisson=self.config.n_min_surface_sampled,
                                                                          cache_dir=self.cache_dir)
        # elif self.sampling_method == 'face_center_plus_poisson':
        #     raise Exception("face_center_plus_poisson has not been implemented yet")
        else:
            raise Exception("Unknown point sampling method provided")

        point_set = points.astype(np.float32)
        point_set, transformations, normals = EHDataset.augment_point_set(point_set,
                                                                           normalize=self.config.normalize,
                                                                           normalize_type=self.config.normalize_type,
                                                                           normalize_scale_factor=self.config.normalize_scale_factor,
                                                                           rotate_axes=self.config.rotate_axes,
                                                                           move_along_axis=self.config.move_along_axis,
                                                                           data_augmentation_jitter=self.config.data_augmentation_jitter,
                                                                           normals=normals)
        self.last_transformations = transformations
        if self.config.point_normals:
            #print("point_set", point_set.shape)
            #print("normals", normals.shape)
            point_set = np.concatenate((point_set, normals.astype(np.float32)), axis=-1)
        point_set = torch.from_numpy(point_set)

        # if self.config.wrap_cls:
        return point_set, np.array([model['cls_idx']]).astype(np.long), transformations

    def __len__(self):
        return len(self.models)

    @staticmethod
    def augment_point_set(point_set,
                          normalize=False,
                          normalize_type='Mean',
                          normalize_scale_factor=1.0,
                          rotate_axes='',
                          move_along_axis='',
                          data_augmentation_jitter=False,
                          normals=None):
        last_transformations = {}

        if normalize:
            point_set, done_translation, done_scale = pc_normalize(point_set,
                                                                   normalize_type,
                                                                   normalize_scale_factor)
            last_transformations = {
                'translation': done_translation,
                'scale': done_scale,
            }

        if 'y' in rotate_axes:
            # randomly rotate around y
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            if normals is not None:
                normals[:, [0, 2]] = normals[:, [0, 2]].dot(rotation_matrix)  # random rotation
            # point_set[:, [3, 5]] = point_set[:, [0, 2]].dot(rotation_matrix)  # rotate normals?
            # self.last_rotations['y'] = rotation_matrix
            if 'rotation' not in last_transformations:
                last_transformations['rotation'] = {}
            last_transformations['rotation']['y'] = rotation_matrix

        if 'z' in rotate_axes:
            # randomly rotate around z
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 1]] = point_set[:, [0, 1]].dot(rotation_matrix)  # random rotation
            if normals is not None:
                normals[:, [0, 1]] = normals[:, [0, 1]].dot(rotation_matrix)  # random rotation
            # last_rotations['z'] = rotation_matrix
            if 'rotation' not in last_transformations:
                last_transformations['rotation'] = {}
            last_transformations['rotation']['z'] = rotation_matrix

        if 'x' in rotate_axes:
            # randomly rotate around x
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [1, 2]] = point_set[:, [1, 2]].dot(rotation_matrix)  # random rotation
            if normals is not None:
                normals[:, [1, 2]] = normals[:, [1, 2]].dot(rotation_matrix)  # random rotation
            # last_rotations['x'] = rotation_matrix
            if 'rotation' not in last_transformations:
                last_transformations['rotation'] = {}
            last_transformations['rotation']['x'] = rotation_matrix

        if 'y' in move_along_axis:
            point_set[:, 1] = point_set[:, 1] + (np.random.normal(0, 0.2, 1)[0] * normalize_scale_factor)

        if data_augmentation_jitter:
            point_set += np.random.normal(0, 0.005, size=point_set.shape)  # random jitter

        return point_set, last_transformations, normals


def pc_normalize(pc, type, normalize_scale_factor=1.0, scale_reference=None):
    assert type in ['mean', 'bottom_aligned']
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))

    done_translation = -centroid
    if type == 'bottom_aligned':
        # move the model up the y-axis so that the all y-vales are positive
        lowest_y = np.min(pc[:, 1], axis=0)
        pc[:, 1] = pc[:, 1] - lowest_y
        done_translation[1] = done_translation[1] - lowest_y

    pc = (pc / m) * normalize_scale_factor
    done_scale = normalize_scale_factor/m

    return pc, done_translation, done_scale