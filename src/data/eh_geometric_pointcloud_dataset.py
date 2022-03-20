import torch
from torch_geometric.data import Data, Dataset
from src.data.eh_pointcloud_dataset import EHDataset


class EHDatasetGeometricProxy(Dataset):
    def __init__(self, root, config, **params):
        self.dataset = EHDataset(root, config, **params)
        self.config = config
        self.models = self.dataset.models
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
        self.last_model = None
        self.last_index = None
        super(EHDatasetGeometricProxy, self).__init__(root, None, None)

    def len(self):
        return self.dataset.__len__()

    def get(self, idx):
        point_set, label = self.dataset.__getitem__(idx)[:2]
        self.last_model = self.dataset.last_model
        self.last_index = self.dataset.last_index
        # points_set => torch tensor [2000, 3] float 32
        #target = torch.zeros(self.num_classes)
        #target[label[0]] = 1.0
        normals = None
        if self.config.point_normals:
            normals = point_set[:, 3:]
        data = Data(x=None, pos=point_set[:, :3], y=torch.tensor(label), normal=normals)
        return data