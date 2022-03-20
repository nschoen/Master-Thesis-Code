import os
import torch
import csv
import open3d as o3d
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh


class ClassificationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.sample_dir = os.path.join(self.root, opt.sample_dir)
        self.dir = os.path.join(opt.dataroot)
        if self.is_stl_extensions(self.sample_dir):
            self.cache_obj(self.sample_dir, "obj-cached")
            self.sample_dir = os.path.join(self.sample_dir, "obj-cached")
        self.classes, self.paths = self.read_models(
            self.dir,
            opt.csv_file,
            self.sample_dir,
            opt.phase,
            opt.cross_validation_set)
        # self.classes, self.class_to_idx = self.find_classes(self.dir)
        # self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase)
        self.models = []
        for p in self.paths:
            self.models.append({'filename': p[0], 'datapath': p[0]})

        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        label = self.paths[index][1]
        mesh = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)
        meta = {'mesh': mesh, 'label': label}
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    # this is when the folders are organized by class...
    @staticmethod
    def read_models(root, csv_file, sample_dir, split='train', cross_validation_set=False):
        models = []
        classes = []

        class_to_idx = {}

        with open(os.path.join(root, csv_file), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                if cross_validation_set and (
                        (split == 'train' and int(row[4]) == cross_validation_set) or (
                        split == 'test' and int(row[4]) != cross_validation_set)
                ) or not cross_validation_set and row[3] != split:
                    continue
                #print("row[3]", row[3])
                #classid_map[int(row[2])] = row[1]
                class_to_idx[row[1]] = int(row[2])
                if row[1] not in classes:
                    classes.append(row[1])

                filename = row[0]
                basename = filename.split('.')[0]

                if not filename.endswith('.obj'):
                    # if given, remove existing extension and add obj extension
                    filename = f"{basename}.obj"

                datapath = os.path.join(sample_dir, filename)
                models.append((datapath, int(row[2])))

        return classes, models

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

    # @staticmethod
    # def make_dataset_by_class(dir, class_to_idx, phase):
    #     meshes = []
    #     dir = os.path.expanduser(dir)
    #     for target in sorted(os.listdir(dir)):
    #         d = os.path.join(dir, target)
    #         if not os.path.isdir(d):
    #             continue
    #         for root, _, fnames in sorted(os.walk(d)):
    #             for fname in sorted(fnames):
    #                 if is_mesh_file(fname) and (root.count(phase)==1):
    #                     path = os.path.join(root, fname)
    #                     item = (path, class_to_idx[target])
    #                     meshes.append(item)
    #     return meshes
