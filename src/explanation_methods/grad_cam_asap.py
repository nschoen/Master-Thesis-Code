import open3d as o3d

from torch.autograd import Variable

import numpy as np
import torch
import types
from src.utils.visualization.heatmap_plotting import create_heatmap_pc, plot_pc_as_mesh


class GradCamAsap:
    def __init__(self, classifier, counterfactual=False, intermedia_layers=False):
        self.classifier = classifier
        self.counterfactual = counterfactual
        self.intermedia_layers = intermedia_layers

        self.features = []
        self.features_xyz = []
        self.gradients = []
        self.classifier_output = -1

        # add save_features hook
        self.classifier.save_features = types.MethodType(self.save_features, self.classifier)

    def save_features(self, ctx, features, pos):
        self.features_xyz.append(pos)
        self.features.append(features)
        features.register_hook(self.save_gradients)

    def save_gradients(self, grad):
        self.gradients.append(grad)

    def calc_cam(self):
        if self.intermedia_layers:
            grads_vals = []
            for grad in self.gradients:
                grads_vals.append(grad.cpu().data.numpy())
        else:
            grads_vals = [self.gradients[-1].cpu().data.numpy()]

        cams = []
        max_val = -1
        min_val = None
        for grads_val in grads_vals:
            if self.counterfactual:
                grads_val = grads_val * -1

            target = self.features[-1]
            target = target.cpu().data.numpy()
            weights = np.mean(grads_val, axis=(0))
            cam = np.zeros(target.shape[0], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * target[:, i]
            cam = np.maximum(cam, 0)  # ReLU
            if np.max(cam) > max_val:
                max_val = np.max(cam)
            if min_val == None or np.min(cam) < min_val:
                min_val = np.min(cam)
            cams.append(cam)
        return cams, max_val, min_val

    def __call__(self, input, target_idx=False):

        # reset values
        self.features = []
        self.features_xyz = []
        self.gradients = []

        output = self.classifier(input)
        self.classifier_output = output
        # self.classifier_output = np.argmax(output.cpu().data.numpy())

        index = target_idx
        if not index:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.to(input.x.device) * output)
        self.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        # Calculate Gradients
        cam_f_list, max_cam_val, min_cam_val = self.calc_cam()
        #cam_f[np.isnan(cam_f)] = 0
        for i in range(len(cam_f_list)):
            cam_f_list[i] = cam_f_list[i] - min_cam_val
            cam_f_list[i] = cam_f_list[i] / max_cam_val

        # interpolate to input
        if self.intermedia_layers:
            pos = []
            for xyz in self.features_xyz:
                pos.append(xyz.data.cpu().numpy())
        else:
            pos = [self.features_xyz[-1].data.cpu().numpy()]

        ref_pds = []
        idx = 0
        for p in pos:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(p)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            ref_pds.append((idx, pcd_tree, p))
            idx += 1

        #if True:
        # visualize point cloud after last pooling
        #    pcd.paint_uniform_color([0, 0, 0])
        #    o3d.visualization.draw_geometries([pcd])

        input_np_flipped = input.pos.data.cpu().numpy()
        cam_input = np.zeros(input_np_flipped.shape[0])

        j = 1
        for i in range(input_np_flipped.shape[0]):
            pt = input_np_flipped[i]

            nearest_point_distance = None
            cam_list = []
            for j, pds, pds_points in ref_pds:
                [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, 1)
                # [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, 7)
                distance = np.linalg.norm(pds_points[idx[0]] - pt)
                if nearest_point_distance is not None and distance > nearest_point_distance:
                    continue
                if nearest_point_distance == None or distance < nearest_point_distance:
                    nearest_point_distance = distance
                    cam_list = []
                cam_list.append(cam_f_list[j][idx[0]])
            cam = 0.0
            for c_val in cam_list:
                if c_val > cam:
                    cam = c_val
            cam_input[i] = c_val # cam_f[idx[0]]
            continue

            # apply rervse distance weighted interpolation
            # http://www.gitta.info/ContiSpatVar/de/html/Interpolatio_learningObject2.html
            weighted_cam_sum = 0
            weight_sum = 0
            min_distance = 0.05  # previous was 0.1
            # if self.progress_mode_input_plus_interpolate_k > 1:
            #     min_distance = (np.linalg.norm(xyz_aggregate_np[idx[1]] - pt) / 2.0).astype('float32')
            for id in idx:
                # d = np.around(max(min_distance, np.linalg.norm(xyz_aggregate_np[id] - pt)), decimals=6)  # min distance 0.1
                d = np.linalg.norm(xyz_aggregate_np[id] - pt)
                if d == 0:
                    weight_sum = cam_f[id]
                    print("distance is 0", str(j))
                    j += 1
                    continue
                d = max(min_distance, d)  # min distance 0.1
                # d = np.linalg.norm(xyz_aggregate_np[id] - pt)
                weighted_cam_sum += (1.0 / d) * cam_f[id]
                weight_sum += (1.0 / d)

            cam_input[i] = weighted_cam_sum / weight_sum

        return cam_input, cam_f_list, self.features_xyz
        # return cam_f

    def set_iteration_index(self, index):
        self.model_index = index

def mesh_grad_cam_asap(
        module,
        classifier,
        dataloader,
        discrete_heatmap=False,
        discrete_heatmap_threshold=0.5,
        intermedia_layers=False,
        show_plain_pc=True):
    grad_cam = GradCamAsap(classifier, intermedia_layers=intermedia_layers)

    for i, data in enumerate(dataloader):
        last_model = dataloader.dataset.models[i]
        data, target = module.get_data(data)
        points = data.pos
        target_idx = target.data.item()
        data.pos = Variable(data.pos)
        data.x = Variable(data.x)
        data.edge_attr = Variable(data.edge_attr)

        mask, features_xyz_cam, features_xyz = grad_cam(data, target_idx=target_idx)
        predicted = np.argmax(grad_cam.classifier_output.cpu().data.numpy())

        print("Filename", last_model['filename'])
        print("predicted", predicted)
        print("target", target_idx)

        if show_plain_pc:
            # Visualize the point cloud first
            prc_r_all = data.pos.contiguous().data.cpu()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(prc_r_all)
            pcd.paint_uniform_color([0, 0, 0])
            o3d.visualization.draw_geometries([pcd])

        heatmap = create_heatmap_pc(points,
                                    mask,
                                    offset=0,
                                    colormap=True,
                                    discrete_heatmap=discrete_heatmap,
                                    discrete_heatmap_threshold=discrete_heatmap_threshold)

        # show heatmap
        o3d.visualization.draw_geometries([heatmap])

