"""
Runs the point dropping experiment with Grad-CAM on PointNet++ model (with point normals) trained on Toy dataset.
The results are saved in ./results/point_dropping.

The plot ./results/point_dropping/gradcam-noHU-113pcds-3000drops-100steps-Accuracy.png is used in the thesis.

Running this script again will create a new results prefixed wiht "gradcam-noHU-SUBMISSION-".
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
    "device": "cuda:1",
    "epochs": 300,
    "workers": 4,
    "batch_size": 64,  # 32 was used during training
    "learning_rate": 0.001,
    "weight_decay_rate": 1e-4,
    "decay_step_size": 40,
    "decay_rate": 0.5,
    "optimizer_type": "Adam",
    "dataset": {
        "dataset": "toy",
        "type": "pointcloud",
        "point_normals": True,
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
model.train()  # required to load correct model
model.point_dropping_experiment(xai_method='gradcam', name='gradcam-noHU-SUBMISSION-', update_cam=False)
