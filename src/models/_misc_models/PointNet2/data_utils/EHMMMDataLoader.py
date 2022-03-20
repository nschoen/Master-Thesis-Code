import numpy as np
import warnings
import os
from torch.utils.data import Dataset

class MMMDataset(Dataset):
    def __init__(self,
                 root,
                 split='train',
                 classes_file='classes.csv',
                 data_augmentation=True,
                 rotate_only_y_axis=False):
        self.models = []
        self.classes = []
        self.classid_map = {}
        self.data_augmentation = data_augmentation
        self.rotate_only_y_axis = rotate_only_y_axis

        with open(os.path.join(root, classes_file), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                if row[3] != split:
                    continue
                if core and row[4] != 'core':
                    continue
                self.models.append({
                    'cls': int(row[2]),
                    # 'datapath': os.path.join(root, row[1], row[0]),
                    'datapath': os.path.join(root, 'data-ply-4000', row[0]),
                    'filename': row[0]
                })
                self.classid_map[int(row[2])] = row[1]

        # create sorted list of classes
        class_ids = list(self.classid_map.keys())
        class_ids.sort()
        self.classes = [self.classid_map[idx] for idx in class_ids]
        print(self.classes)


    def __getitem__(self, index):
        model = self.models[index]

        points, normals = pointcloud_utils.read_point_cloud_poisson(model['datapath'], n=self.npoints)
        point_set = points.astype(np.float32)

        if self.data_augmentation:
            # randomly rotate around y
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            # point_set[:, [3, 5]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation

            # if self.rotate_only_y_axis == False:
            #     # randomly rotate around z
            #     theta = np.random.uniform(0, np.pi * 2)
            #     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            #     point_set[:, [0, 1]] = point_set[:, [0, 1]].dot(rotation_matrix)  # random rotation
            #     # randomly rotate around x
            #     theta = np.random.uniform(0, np.pi * 2)
            #     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            #     point_set[:, [1, 2]] = point_set[:, [1, 2]].dot(rotation_matrix)  # random rotation

            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set)

        return point_set, model['cls']

    def __len__(self):
        return len(self.models)        print(label.shape)