import open3d as o3d
import numpy as np
import torch
from torch_geometric.data import DataLoader, Data
import copy
import time
import multiprocessing
from src.utils.visualization.heatmap_plotting import create_heatmap_pc
from src.explanation_methods.rise import RISE

class MeshRISE(RISE):
    def __init__(self, model, gpu_batch=8):
        super(MeshRISE, self).__init__(model, gpu_batch, type='mesh')
        print("self.gpu_batch", self.gpu_batch)

    def forward(self, data, remove_edges=False):
        device = data.pos.device
        M = self.M
        N = data.pos.shape[0]

        weights = self.masks.sum(0)
        p = []

        data = Data(pos=data.pos,
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    y=data.y,
                    vertices=data.vertices,
                    neighbour_ids=data.neighbour_ids,
                    edges=data.edges,
                    normal=data.normal).to('cpu')

        # build vertices => edge_id map first to now to which edges are connected to the key vertice
        vertice_to_edge = {} # vertice => [edge_id]
        data_cp = copy.copy(data)
        for e_id in range(len(data_cp.edge_index[0])):
            for vert in [data_cp.edge_index[0, e_id], data_cp.edge_index[1, e_id]]:
                vert = vert.item()
                if vert not in vertice_to_edge:
                    vertice_to_edge[vert] = []
                if e_id not in vertice_to_edge[vert]:
                    vertice_to_edge[vert].append(e_id)

        pool = multiprocessing.Pool(64)
        jobs = []

        # data is a python object not a numpy array, cannot simple parralelize masking, have to iterate
        batch_list = []
        data_list = []
        for i in range(M):
            start = time.time()

            mask = self.masks[i]

            # alternative 1:
            data_cp = copy.copy(data)
            data_cp.x = data_cp.x * mask.unsqueeze(-1)  # mask x
            data_cp.pos = data_cp.pos * mask.unsqueeze(-1)  # mask pos
            # mask edges
            if remove_edges:
                # new_edge_index = [[], []]
                # new_edge_attr = []
                # remove_idx = np.nonzero(mask == 0)
                # for e_id in range(len(data_cp.edge_index[0])):
                #     from_edge_id = data_cp.edge_index[0, e_id]
                #     to_edge_id = data_cp.edge_index[1, e_id]
                #     if from_edge_id not in remove_idx and to_edge_id not in remove_idx:
                #         new_edge_index[0].append(from_edge_id)
                #         new_edge_index[1].append(to_edge_id)
                #         new_edge_attr.append(data_cp.edge_attr[e_id])
                # data_cp.edge_index = torch.tensor(new_edge_index, dtype=torch.long)
                # data_cp.edge_attr = torch.tensor(new_edge_attr, dtype=torch.float32)

                # 250x faster solution:
                remove_idx = np.nonzero(mask == 0)
                edge_mask = np.ones(data_cp.edge_attr.shape[0], dtype=np.bool)
                for rem_id in remove_idx:
                    for e_id in vertice_to_edge[rem_id.item()]:
                        edge_mask[e_id] = False
                data_cp.edge_attr = data_cp.edge_attr[edge_mask]
                data_cp.edge_index = data_cp.edge_index.T[edge_mask].T
            else:
                # mask edge attributes of all edges that are going from or to a masked node
                remove_idx = np.nonzero(mask == 0)
                for rem_id in remove_idx:
                    for e_id in vertice_to_edge[rem_id.item()]:
                        data_cp.edge_attr[e_id] = data_cp.edge_attr[e_id] * 0
                #for e_id in range(len(data_cp.edge_index[0])):
                #    from_vert_id = data_cp.edge_index[0, e_id]
                #    to_vert_id = data_cp.edge_index[1, e_id]
                #    if from_vert_id in remove_idx or to_vert_id in remove_idx:
                #        data_cp.edge_attr[e_id] = data_cp.edge_attr[e_id] * 0
            #data_cp.batch = i % self.gpu_batch
            data_cp = Data(pos=data_cp.pos.type(torch.float32),
                           x=data_cp.x.type(torch.float32),
                           edge_index=data_cp.edge_index,
                           edge_attr=data_cp.edge_attr.type(torch.float32),
                           y=data_cp.y.type(torch.long),
                           vertices=data_cp.vertices,
                           neighbour_ids=data_cp.neighbour_ids,
                           edges=data_cp.edges,
                           normal=data_cp.normal)
            data_list.append(data_cp)

            # print("time for preprocessing", time.time()-start)

            # alternative 2:
            # job = pool.apply_async(mask_and_copy, (data, mask, vertice_to_edge, remove_edges,))
            # jobs.append(job)

            #if len(jobs) < self.gpu_batch and i < M-1: # alternative 2
            if len(data_list) < self.gpu_batch and i < M-1:  # alternative
                continue

            if i % 100 == 0:
                print("batch start", i)
                print("total time for preprocessing", time.time()-start)

            loader = DataLoader(data_list, batch_size=self.gpu_batch, shuffle=False)
            data_list = []  # reset

            for masked_batch in loader:
                output = self.model(masked_batch.to(device))
                output = output[0] if type(output) == tuple else output
                confidence = torch.nn.functional.softmax(output)
                p.append(confidence.clone().detach().cpu())
                del output
                del confidence
                torch.cuda.empty_cache()

        p = torch.cat(p)
        sal = torch.matmul(p.data.transpose(0, 1).double(), self.masks.double())
        sal = sal.data / weights

        # normalize
        for dim1 in range(sal.shape[0]):
            sal[dim1] = sal[dim1] - sal[dim1].min()
            sal[dim1] = sal[dim1] / sal[dim1].max()

        return sal


def show_mesh_rise_explanation(module,
                               root_dir,
                               classifier,
                               loader,
                               dataset,
                               discrete_heatmap=True,
                               discrete_heatmap_threshold=0.5,
                               show_plain_pc=True):
    for i, data in enumerate(loader):
        data, target = module.get_data(data)
        last_model = dataset.models[i]

        if show_plain_pc:
            points = np.asarray(data.pos.cpu())
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([0, 0, 0])
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, mesh_frame])

        # get prediction
        output = classifier(data)
        pred = output[0] if type(output) == tuple else output
        predicted = pred.max(dim=1)[1].cpu().item()

        rise = MeshRISE(classifier, module.config.batch_size)
        s = [16, 32]
        s_id = "-".join([str(sx) for sx in s])
        rise.generate_masks(data.pos.cpu(),
                            2000,  # 3000,
                            p1=0.1,
                            s=s,
                            cache_name=dataset.models[i]['filename'].split('.')[0] + s_id,
                            root_dir=root_dir)

        exp = rise.forward(data, remove_edges=True)

        print("Filename", last_model['filename'])
        print("predicted", predicted)
        print("target", target[0])

        heatmaps = []
        for id in range(exp.shape[0]):
            hmap = create_heatmap_pc(data.pos,
                                     exp[id].cpu().data,
                                     offset=id,
                                     colormap=True,
                                     discrete_heatmap=discrete_heatmap,
                                     discrete_heatmap_threshold=discrete_heatmap_threshold)
            heatmaps.append(hmap)

        target_id = target[0].item()
        print("target heatmap id", target_id)
        o3d.visualization.draw_geometries([heatmaps[target_id]])
        if target_id != predicted:
            print("predicted heatmap id", predicted)
            o3d.visualization.draw_geometries([heatmaps[predicted]])

        # show all heatmaps
        o3d.visualization.draw_geometries(heatmaps)