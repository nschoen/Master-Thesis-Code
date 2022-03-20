"""
Final PointNet model (without point normals) trained on Toy dataset.
"""

# required to allow calling this script directly via console (pycharm automatically detects root)
from os.path import dirname, join, abspath
import sys
sys.path.insert(1, abspath(join(dirname(__file__), '../../../../..')))

from src.model_loader import load_model

config = {
    "model": "PointNet",
    "device": "cpu",
    "epochs": 400,
    "workers": 4,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay_rate": 1e-4,
    "decay_step_size": 40,
    "decay_rate": 0.5,
    "optimizer_type": "Adam",
    "pointnet_feature_transform": True,
    "dataset": {
        "dataset": "toy",
        "type": "pointcloud",
        "point_normals": False,
        "models_csv": "toy.csv",
        "point_sampling_method": "poisson",
        "mesh_dir": "data",
        "n_points": 3000,
        "data_augmentation_jitter": True,
        "rotate_axes": "y",
        "move_along_axis": "",
        "normalize": True,
        "normalize_type": "mean",
        "normalize_scale_factor": 1.0,
    },
}

model = load_model(config)
model.train()
model.evaluate()
