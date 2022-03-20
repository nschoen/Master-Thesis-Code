import open3d as o3d
import torch
import numpy as np
import types
from torch.autograd import Variable
from src.utils.visualization.heatmap_plotting import create_heatmap_pc, plot_pc_as_mesh

gradient = None
features = None


def save_gradient(grad):
    global gradient
    gradient = grad


# register hook
def hook_pre_max(feat_self, x):
    global features
    features = x
    x.register_hook(save_gradient)


class GradCamPointNet1:
    def __init__(self, model, counterfactual=False, normalize=True):
        self.model = model
        self.counterfactual = counterfactual
        self.normalize = normalize
        self.gradient = None

    def __call__(self, input, index=None):
        global gradient
        global features

        device = input.device

        feats = self.model.feat
        feats.hook_pre_max = types.MethodType(hook_pre_max, feats)

        output, _, _ = self.model(input.to(device))
        self.classifier_output = output
        #self.classifier_output =  np.argmax(output.cpu().data.numpy())

        # features, output = self.extractor(input.cuda())
        # self.classifier_output = output

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.to(device) * output)

        self.model.feat.zero_grad()
        self.model.zero_grad()

        one_hot.backward(retain_graph=True)

        grads_val = gradient.cpu().data.numpy()
        if self.counterfactual:
            grads_val = grads_val * -1

        target = features # features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :]

        cam = np.maximum(cam, 0)  # ReLU

        if self.normalize:
            #cam = cv2.resize(cam, (224, 224))
            cam = cam - np.min(cam)
            if np.max(cam) > 0:
                # in the drop point experiment the gradients are at some time 0 leading to a division by zero error
                cam = cam / np.max(cam)

        return cam, None, None

    def set_iteration_index(self, index):
        self.model_index = index


def show_point_grad_cam_pn1_explanation(module,
                                        classifier,
                                        dataloader,
                                        discrete_heatmap=False,
                                        discrete_heatmap_threshold=0.5,
                                        show_plain_visualiation=True,
                                        visualize_mesh=True):
    grad_cam = GradCamPointNet1(model=classifier, counterfactual=False)
    dataset = dataloader.dataset

    for i, data in enumerate(dataloader):
        transformations = data[2]
        points, target = module.get_data(data)
        points_xyz = points[0, :, :3]  # => [N, 3] remove normals
        target_idx = target[:, 0].cpu().item()

        mask, _, _ = grad_cam(points, target_idx)
        predicted = np.argmax(grad_cam.classifier_output.cpu().data.numpy())

        print("========")
        print("Filename", dataset.models[i]['filename'])
        print("Predicted class ID", predicted)
        print("Target class ID", target_idx)

        if show_plain_visualiation:
            # Visualize the point cloud first
            print("Showing plain input point cloud")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_xyz.cpu().data.numpy())
            pcd.paint_uniform_color([0, 0, 0])
            o3d.visualization.draw_geometries([pcd])

        heatmap = create_heatmap_pc(points_xyz,
                                    mask,
                                    offset=0,
                                    colormap=True,
                                    discrete_heatmap=discrete_heatmap,
                                    discrete_heatmap_threshold=discrete_heatmap_threshold)

        # show heatmap
        print("Showing heatmap for target class")
        o3d.visualization.draw_geometries([heatmap])

        if visualize_mesh:
            print("Showing heatmap for target class mapped onto the original mesh")
            plot_pc_as_mesh(heatmap, dataset.models[i]['datapath'], transformations)
