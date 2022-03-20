import open3d as o3d
from torch.autograd import Variable
import numpy as np
import torch
import types
from src.utils.visualization.heatmap_plotting import create_heatmap_pc, plot_pc_as_mesh


class GradCamPointnet2:
    def __init__(self, classifier, counterfactual=False):
        self.classifier = classifier
        self.counterfactual = counterfactual

        self.features = None
        self.features_xyz = None
        self.gradients = None
        self.classifier_output = -1

        sa3 = self.classifier.sa3
        sa3.save_pre_max_features = types.MethodType(self.save_pre_max_features, sa3)

    def save_pre_max_features(self, ctx, xyz, features):
        self.features_xyz = xyz
        self.features = features
        features.register_hook(self.save_gradients)

    def save_gradients(self, grad):
        self.gradients = grad

    def calc_cam(self):
        grads_val = self.gradients.cpu().data.numpy()
        if self.counterfactual:
            grads_val = grads_val * -1

        target = self.features
        target = target.cpu().data.numpy()[0, :]
        weights = np.mean(grads_val, axis=(2))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :]
        return np.maximum(cam, 0)  # ReLU

    def __call__(self, input, target_idx=False):
        """
        Input:
            input: [B, N, C]
        """
        device = input.device

        # reset values
        self.features = None
        self.features_xyz = None
        self.gradients = None

        output, _ = self.classifier(input)
        self.classifier_output = output
        # self.classifier_output = np.argmax(output.cpu().data.numpy())

        index = target_idx
        if not index:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.to(device) * output)
        self.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        # Calculate Gradients
        cam_f = self.calc_cam()
        #cam_f[np.isnan(cam_f)] = 0
        cam_f = cam_f - np.min(cam_f)
        #try:
        cam_f = cam_f / np.max(cam_f)
        #except:
        #    # la = np.max(cam_f)

        # interpolate to input
        pcd = o3d.geometry.PointCloud()
        xyz_aggregate_np = self.features_xyz.data.cpu().numpy()[0]
        pcd.points = o3d.utility.Vector3dVector(xyz_aggregate_np[:, 0:3])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        input_points = input.data.cpu().numpy()[0]
        cam_input = np.zeros(input_points.shape[0])

        j = 1
        for i in range(input_points.shape[0]):
            pt = input_points[i][0:3]
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, 1)
            # [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, 7)

            cam_input[i] = cam_f[idx[0]]
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

        return cam_input, cam_f, self.features_xyz

    def set_iteration_index(self, index):
        self.model_index = index


def show_point_grad_cam_pn2_explanation(
        module,
        classifier,
        data_loader,
        visualize_mesh=False,
        discrete_heatmap=False,
        discrete_heatmap_threshold=0.5,
        show_plain_pc=True,
        explain_class_id=None):
    grad_cam = GradCamPointnet2(classifier)

    for i, data in enumerate(data_loader):
        last_model = data_loader.dataset.models[i]
        transformations = data[2]
        points, target = module.get_data(data)
        target_idx = target[0].data.cpu().item()
        points_xyz = points[0, :, :3].data.cpu()

        mask, _, _ = grad_cam(points, target_idx=(explain_class_id if explain_class_id else target_idx))
        predicted = np.argmax(grad_cam.classifier_output.cpu().data.numpy())

        print("Filename", last_model['filename'])
        print("predicted", predicted)
        print("target", target_idx)
        if explain_class_id:
            print("showing heatmap for class id", explain_class_id)

        if show_plain_pc:
            # Visualize the point cloud first
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_xyz)
            pcd.paint_uniform_color([0, 0, 0])
            o3d.visualization.draw_geometries([pcd])

        heatmap = create_heatmap_pc(points_xyz,
                                    mask,
                                    offset=0,
                                    colormap=True,
                                    discrete_heatmap=discrete_heatmap,
                                    discrete_heatmap_threshold=discrete_heatmap_threshold)

        # show heatmap
        o3d.visualization.draw_geometries([heatmap])

        if visualize_mesh:
            plot_pc_as_mesh(heatmap, last_model['datapath'], transformations)