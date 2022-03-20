import os
import sys
sys.path.append('../../../../../')
from src.model_loader import load_model
from src.base_model import BaseModel

# config = {
#     "dataset": {
#         "dataset": "toy",
#         "type": "pointcloud",
#         "point_normals": False,
#         "models_csv": "toy.csv",
#         "point_sampling_method": "poisson",
#         "mesh_dir": "data",
#         "n_points": 3000,
#         "data_augmentation_jitter": True,
#         "rotate_axes": "y",
#         "move_along_axis": "",
#         "normalize": True,
#         "normalize_type": "mean",
#         "normalize_scale_factor": 1.0,
#     },
# }

config = {
    "model": "MeshCNN",
    "device": "cuda:1",
    #"epochs": 300,
    #"batch_size": 32,
    #"learning_rate": 0.001,
    #"weight_decay_rate": 1e-4,
    #"decay_step_size": 40,
    #"decay_rate": 0.5,
    #"optimizer_type": "Adam",
    "model_opt": {
        "gpu_ids": [1],
        "dataroot": os.path.join(BaseModel.DATASETS_DIR, "toy"),
        "csv_file": "toy_meshcnn.csv",
        "sample_dir": "data",
        "name": "toy",
        "ncf": [64, 128, 256, 256],
        "pool_res": [15000, 9000, 6000, 3000],
        #"pool_res": [8000, 7500, 7000, 6500],
        "ninput_edges": 20000,
        #"ninput_edges": 18000,
        "norm": 'group',
        "resblocks": 1,
        "flip_edges": 0.2,
        "slide_verts": 0.2,
        "num_aug": 20,
        "niter_decay": 100,
        "epoch_count": 250,
    }
}

model = load_model(config)
model.train()
model.evaluate()
