"""
Final EdgeGlobal model trained on EH PRO dataset.

Note: Since EH PRO is IP of Endress+Hauser it has not been shared. Thus, this model cannot be executed.
"""

# required to allow calling this script directly via console (pycharm automatically detects root)
from os.path import dirname, join, abspath
import sys
sys.path.insert(1, abspath(join(dirname(__file__), '../../../../..')))

from src.model_loader import load_model
from torch_geometric.transforms import Compose, NormalizeScale, RandomRotate, RandomTranslate, AddSelfLoops

config = {
    "model": "EdgeGlobal",
    "device": "cuda:1",  # cuda:1
    "epochs": 230,
    "workers": 4,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay_rate": 1e-3,
    "decay_step_size": 20,
    "decay_rate": 0.7,
    "asap_use_norm": True,
    "asap_use_absolute_pos": True,
    "asap_normalize_direction": False,
    #"asap_num_layers": 2,
    "dataset": {
        #"shuffle": False,
        "dataset": "eh_pro",
        "type": "mesh_edge_unit",
        "models_csv": "classes.csv",
        #"mesh_dir": "data-2000",
        "mesh_dir": ["data-2000", "data-2200", "data-2400", "data-2600"],
        "test_mesh_dir": ["data-2400"],
        "num_aug": 20,
        "scale_verts": True,
        "flip_edges": 0.0,
        #"flip_edges": 0.2,
        "slide_verts": 0.2,
        "mesh_use_xyz": True,
        "mesh_edge_normals": True,
        "self_loop": False,
        "transform": Compose([NormalizeScale(), RandomRotate(degrees=(-180, 180), axis=1), RandomTranslate(0.005)]),  # AddSelfLoops
        "test_transform": Compose([NormalizeScale()]), # AddSelfLoops
        #NormalizeFeatures
    },
}

model = load_model(config)
model.train()
model.evaluate()
