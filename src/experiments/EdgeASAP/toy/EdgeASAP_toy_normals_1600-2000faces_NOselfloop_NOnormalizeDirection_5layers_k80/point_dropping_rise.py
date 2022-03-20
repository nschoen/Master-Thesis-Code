"""
Executes RISE on the final EdgeASAP model trained on Toy dataset.

It uses a few demo samples of the folder <root>/src/explanation_methods/samples/toy-1600
"""

# required to allow calling this script directly via console (pycharm automatically detects root)
from os.path import dirname, join, abspath
import sys
sys.path.insert(1, abspath(join(dirname(__file__), '../../../../..')))

from src.model_loader import load_model
from torch_geometric.transforms import Compose, NormalizeScale, RandomRotate, RandomTranslate, AddSelfLoops

config = {
    "model": "asap_pos_config",
    "device": "cuda:1",
    "epochs": 219,
    "workers": 4,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay_rate": 1e-3,
    "decay_step_size": 20,
    "decay_rate": 0.7,
    "asap_num_layers": 5,
    "asap_cluster_ratio": 0.8,
    "asap_hidden_neurons": 64,
    "asap_use_norm": True,
    "asap_use_absolute_pos": True,
    "asap_normalize_direction": False,
    "dataset": {
        "dataset": "toy",
        "type": "mesh_edge_unit",
        "models_csv": "toy.csv",
        "mesh_dir": ["data-1600", "data-1800", "data-2000"],
        "test_mesh_dir": ["data-1600"],
        "num_aug": 20,
        "scale_verts": True,
        "flip_edges": 0.0,
        "slide_verts": 0.2,
        "mesh_use_xyz": True,
        "mesh_edge_normals": True,
        "self_loop": False,
        "transform": Compose([NormalizeScale(), RandomRotate(degrees=(-180, 180), axis=1), RandomTranslate(0.005)]),
        "test_transform": Compose([NormalizeScale()]),  # removed AddSelfLoops
    },
}

model = load_model(config)
model.train()
model.point_dropping_experiment(xai_method='rise',
                                name='rise-noHU-',
                                update_cam=False,
                                num_drops=250,
                                steps=50)