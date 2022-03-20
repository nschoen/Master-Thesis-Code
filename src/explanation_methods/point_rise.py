import open3d as o3d
import numpy as np
import torch
from torch.autograd import Variable
from random import randint
from src.explanation_methods.rise import RISE
from src.utils.visualization.heatmap_plotting import create_heatmap_pc, plot_pc_as_mesh


class PointRISE(RISE):
    def __init__(self, model, gpu_batch=16):
        super(PointRISE, self).__init__(model, gpu_batch=gpu_batch, type='pointcloud')

    def forward(self, x):
        """
        Input:
            x: point cloud, [N, C]
        """
        M = self.M
        N = x.shape[0]
        stack = torch.mul(x.unsqueeze(0).data.repeat([M, 1, 1]).double(), self.masks.view(M, N, 1))
        self.stack = stack

        weights = self.masks.sum(0)

        p = []
        for i in range(0, M, self.gpu_batch):
            output = self.model(stack[i:min(i + self.gpu_batch, M)].float())
            confidence = torch.nn.functional.softmax(output[0])
            p.append(confidence.clone().detach())
            del output
            del confidence
            torch.cuda.empty_cache()

        p = torch.cat(p)
        sal = torch.matmul(p.data.transpose(0, 1).double(), self.masks.double())
        sal = sal / weights

        # normalize
        for dim1 in range(sal.shape[0]):
            sal[dim1] = sal[dim1] - sal[dim1].min()
            sal[dim1] = sal[dim1] / sal[dim1].max()

        return sal


def show_point_rise_explanation(
        module,
        root_dir,
        classifier,
        dataloader,
        visualize_mesh=True,
        discrete_heatmap=False,
        discrete_heatmap_threshold=0.5,
        show_plain_pc=True,
        show_random_mask_inputs=False,
        explain_class_id=None):
    dataset = dataloader.dataset

    for i, data in enumerate(dataloader):
        transformations = data[2]
        points, target = module.get_data(data)
        points = points[0]
        points_xyz = points[:, 0:3]  # => [N, 3] only use xyz (remove normals)

        output = classifier(points.unsqueeze(0))  # requires [B=1, N, C]
        predicted = output[0].data.max(1)[1].cpu().item()

        target_id = data[1][0, 0].item()

        print("========")
        print("Filename", dataset.models[i]['filename'])
        print("Predicted class ID", predicted)
        print("Target class ID", target_id)
        if explain_class_id:
            print("showing heatmap for class id", explain_class_id)

        if show_plain_pc:
            # Visualize the point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_xyz.cpu().data.numpy())
            pcd.paint_uniform_color([0, 0, 0])
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, mesh_frame])

        print("Creating heatmap ...")
        rise = PointRISE(classifier)
        rise.generate_masks(points_xyz,
                            3000,
                            s=[16, 32, 64],
                            cache_name=dataset.models[i]['filename'].split('.')[0],
                            root_dir=root_dir)

        exp = rise.forward(points)
        print("Done creating heatmap")

        if show_random_mask_inputs:
            # used in thesis for visualization of the RISE method
            # plot 5 random masked input
            for mi in range(5):
                rand_id = randint(0, 3000)
                mask = rise.masks[rand_id].cpu().data.numpy()
                mask_points = np.asarray(pcd.points)
                mask_points = (mask_points.T * mask).T
                pcd_mask = o3d.geometry.PointCloud()
                pcd_mask.points = o3d.utility.Vector3dVector(mask_points)
                pcd_mask.paint_uniform_color([0, 0, 0])
                o3d.visualization.draw_geometries([pcd_mask])

        heatmaps = []
        for id in range(exp.shape[0]):
            hmap = create_heatmap_pc(points,
                                     exp[id].cpu().data,
                                     offset=id,
                                     colormap=True,
                                     discrete_heatmap=discrete_heatmap,
                                     discrete_heatmap_threshold=discrete_heatmap_threshold)
            heatmaps.append(hmap)

        if explain_class_id:
            print("Showing heatmap of (configured) class ID", explain_class_id)
            o3d.visualization.draw_geometries([heatmaps[explain_class_id]])

        print("Showing heatmap of target class ID", target_id)
        o3d.visualization.draw_geometries([heatmaps[target_id]])
        if target_id != predicted:
            print("Showing heatmap of predicted class ID", predicted)
            o3d.visualization.draw_geometries([heatmaps[predicted]])

        # show all heatmaps
        print("Showing heatmaps for all class IDs (from left to right)")
        o3d.visualization.draw_geometries(heatmaps)

        if visualize_mesh:
            print("Showing target class heatmap mapped back to the original mesh")
            # recreate heatmap with offset=0
            hm = create_heatmap_pc(points,
                                   exp[target_id].cpu().data,
                                   offset=0,
                                   colormap=True,
                                   discrete_heatmap=discrete_heatmap,
                                   discrete_heatmap_threshold=discrete_heatmap_threshold)
            plot_pc_as_mesh(hm, dataset.models[i]['datapath'], transformations)
