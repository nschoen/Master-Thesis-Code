import os
import sys
import torch
import importlib
import provider
from src.base_model import BaseModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

class Model(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        assert self.config.pn2_model_type in ['pointnet2_cls_msg', 'pointnet2_cls_ssg', 'pointnet2_cls_msg_custom', 'pointnet2_cls_msg_face_unit']

    def get_model(self, num_classes=4):
        print("self.config.pn2_model_type", self.config.pn2_model_type)
        print("self.config.pn2_first_SA_n_sample_factor", self.config.pn2_first_SA_n_sample_factor)

        normal_channel = True if hasattr(self.config.dataset, 'point_normals') and self.config.dataset.point_normals else False
        print("normal_channel", normal_channel)

        MODEL = importlib.import_module(self.config.pn2_model_type)
        self.model = MODEL.get_model(num_classes,
                                     normal_channel=normal_channel,
                                     first_SA_n_sample_factor=self.config.pn2_first_SA_n_sample_factor)
        self.criterion = MODEL.get_loss().to(self.config.device)

    def loss_criterion(self, predicted, target, complete_output=None):
        return self.criterion(predicted, target.long(), complete_output[1])

    def get_data(self, data):
        points, target = data[:2]
        #points = points.data.numpy()
        #points = provider.random_point_dropout(points)
        #points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        #points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        #points = torch.tensor(points)
        target = target[:, 0]
        #points = points.transpose(2, 1)
        return points.to(self.config.device), target.to(self.config.device)


