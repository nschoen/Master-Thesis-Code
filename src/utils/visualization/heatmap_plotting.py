import open3d as o3d
import numpy as np
import cv2

def create_heatmap_pc(input,
                      mask,
                      offset=0,
                      colormap=True,
                      discrete_heatmap=True,
                      discrete_heatmap_threshold=0.5):
    """
    Input:
        input: point cloud, [N, 3]
        mask: target points, [N]
        offset: index of object when multiple heatmaps are rendered together
        colormap: create blur/red colors for continuous mask values between 0 and 1 or
    Output:
        heatmap: o3d.geometry.PointCloud
    """
    N = input.size(0)
    points_xyz = input[:, :3].data.cpu()
    points_xyz[:, 0] += offset * 2.1

    if discrete_heatmap:
        # discretize mask into two bins
        mask = np.digitize(mask, bins=[discrete_heatmap_threshold])
        mask = np.clip(mask, 0.15, 0.9)

    if colormap is True:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    else:
        heatmap = mask
    heatmap = np.float32(heatmap) / 255
    heatmap = np.squeeze(heatmap, axis=1)
    heatmap = np.squeeze(heatmap)  # , axis=1
    heatmap[:, [0, 1, 2]] = heatmap[:, [2, 1, 0]]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.paint_uniform_color([0, 0, 0])
    colors = np.asarray(pcd.colors)
    for j in range(N):
        colors[j] = [heatmap[j, 0], heatmap[j, 1], heatmap[j, 2]]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def plot_pc_as_mesh(pcd, stl_path, transformations):
    # load stl into a mesh
    mesh = o3d.io.read_triangle_mesh(stl_path)

    # scale mesh to size of scaled point cloud
    new_vertices = np.asarray(mesh.vertices)
    if 'translation' in transformations:
        new_vertices = new_vertices + transformations['translation'].data.numpy()
    if 'rotation' in transformations:
        if 'y' in transformations['rotation']:
            new_vertices[:, [0, 2]] = new_vertices[:, [0, 2]].dot(
                transformations['rotation']['y'].data.numpy())
        if 'z' in transformations['rotation']:
            new_vertices[:, [0, 1]] = new_vertices[:, [0, 1]].dot(
                transformations['rotation']['z'].data.numpy())
        if 'x' in transformations['rotation']:
            new_vertices[:, [1, 2]] = new_vertices[:, [1, 2]].dot(
                transformations['rotation']['x'].data.numpy())
    if 'scale' in transformations:
        new_vertices = new_vertices * transformations['scale'].data.numpy()

    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)

    # samples more vertices by supdividing triangles in 4 triangles
    # reason is to have more points/vetices to color => more fine grained
    mesh = mesh.subdivide_midpoint(1)

    # color vertices
    vertices = mesh.vertices
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    vertex_colors = np.zeros((np.asarray(vertices).shape[0], 3)).astype(np.float64)

    for i in range(len(vertices)):
        pt = vertices[i]
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, 1)
        vertex_colors[i] = pcd.colors[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([mesh, pcd, mesh_frame])
