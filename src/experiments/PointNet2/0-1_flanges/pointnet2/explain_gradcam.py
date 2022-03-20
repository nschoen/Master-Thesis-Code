"""
Explain PointNet++ on 0/1 Flange Dataset with Grad-CAM

It uses a few demo samples of the folder <root>/src/explanation_methods/samples/0-1_flanges
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
    "epochs": 62,
    "workers": 4,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay_rate": 1e-4,
    "decay_step_size": 40,
    "decay_rate": 0.5,
    "optimizer_type": "Adam",
    "dataset": {
        "dataset": "eh_mmm_0-1_flanges",
        "type": "pointcloud",
        "models_csv": "classes-custom-flanges-1-hole-axis.csv",
        "point_sampling_method": "poisson",
        "mesh_dir": "data",
        "n_points": 4000,
        "data_augmentation_jitter": True,
        "rotate_axes": "",
        "move_along_axis": "",
        "normalize": True,
        "normalize_type": "mean",
        "normalize_scale_factor": 1.0,
    },
}

model = load_model(config)
model.train()
model.explain_samples("0-1_flanges", xai_method='gradcam', explain_class_id=0)
model.explain_samples("0-1_flanges", xai_method='gradcam', explain_class_id=1)
