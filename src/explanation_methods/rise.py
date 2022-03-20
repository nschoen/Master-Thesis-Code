import torch.nn as nn
import math
import numpy as np
import os
import torch
import abc
from src.models.PointNet2.models.pointnet_util import farthest_point_sample, index_points, square_distance
from tqdm import tqdm


class RISE(nn.Module):
    def __init__(self, model, gpu_batch=16, type='pointcloud'):
        super(RISE, self).__init__()
        self.model = model
        self.model.eval()
        self.gpu_batch = gpu_batch
        self.type = type

    def generate_masks(self, xyz, M=3000, s=[16, 32, 64], p1=0.15, cache_name='', root_dir=''):
        """
        Input:
            xyz: xyz points or vertices, [N, C]
            M: number of masks, int
        """
        xyz = xyz[:, 0:3]
        xyz_unsqueezed = xyz.unsqueeze(0)
        N, C = xyz.shape
        device = xyz.device

        # load cached masks
        cache_root_dir = os.path.join(root_dir, 'tmp', f"rise_masks_{self.type}")
        if not os.path.exists(cache_root_dir):
            os.makedirs(cache_root_dir)
        cache_path = os.path.join(cache_root_dir, f"{cache_name}-{M}masks-{N}points-{p1}p1.npy")
        if os.path.exists(cache_path):
            self.load_masks(device, cache_path)
            return

        # sample N masks
        masks = np.zeros((M, N))

        # multiprocessing can lead to problems. Just sequentially create masks:
        # TODO: parallel the mask generation
        for i in tqdm(range(M), desc='Generating filters'):
            n_sample = s[math.floor(i / (M / len(s)))]
            fps_idx = farthest_point_sample(xyz_unsqueezed, n_sample)
            cluster_activations = np.random.rand(n_sample) > p1
            cluster_activations = cluster_activations.astype('float32')
            torch.cuda.empty_cache()
            new_xyz = index_points(xyz_unsqueezed, fps_idx)

            _, S, _ = new_xyz.shape

            sqrdists = square_distance(xyz_unsqueezed, new_xyz)
            cluster_assignment = torch.argmin(sqrdists, 2)[0]

            cluster_activations = np.repeat(cluster_activations.reshape(1, n_sample), N, axis=0)
            ca = cluster_assignment.cpu()
            masks[i] = cluster_activations[np.arange(N), ca[:]]

        # cache results
        np.save(cache_path, masks)

        # set class variables M and generated Masks
        self.M = M
        self.masks = torch.from_numpy(masks).double().to(device)

    def load_masks(self, device, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).double().to(device)
        self.M = self.masks.shape[0]

    @abc.abstractmethod
    def forward(self, x):
        pass
