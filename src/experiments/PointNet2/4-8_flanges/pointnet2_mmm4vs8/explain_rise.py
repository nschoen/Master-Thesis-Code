"""
Explain PointNet++ on 4/8 Flange Dataset with RISE

It uses a few demo samples of the folder <root>/src/explanation_methods/samples/4-8_flanges
"""

# required to allow calling this script directly via console (pycharm automatically detects root)
from os.path import dirname, join, abspath
import sys
sys.path.insert(1, abspath(join(dirname(__file__), '../../../../..')))

from src.model_loader import load_model

config = {
    "model": "PointNet2",
    "pn2_model_type": "pointnet2_cls_msg",
    "pn2_first_SA_n_sample_factor": 2,
    "device": "cuda:0",
    "epochs": 200,
    "workers": 8,
    "batch_size": 24,
    "learning_rate": 0.001,
    "weight_decay_rate": 1e-4,
    "decay_step_size": 20,
    "decay_rate": 0.7,
    "optimizer_type": "Adam",
    "dataset": {
        "dataset": "eh_mmm",
        "models_csv": "classes-4-vs-8-drills-simple.csv",
        "type": "pointcloud",
        "point_sampling_method": "poisson",
        "mesh_dir": "data",
        "n_points": 4000,
        "data_augmentation_jitter": True,
        "rotate_axes": "y",
        "move_along_axis": "y",
        "normalize": True,
        "normalize_type": "mean",
        "normalize_scale_factor": 1.0,
    },
}

model = load_model(config)
model.train()  # required to load correct model
model.explain_samples("4-8_flanges", xai_method='rise', discrete_heatmap=False)
