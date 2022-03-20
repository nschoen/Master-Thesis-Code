import open3d as o3d
from src.utils.pointcloud_utils.pointcloud_utils import create_point_cloud_from_file, sample_nodes_and_poisson_from_mesh, sample_poisson_from_mesh
import numpy as np


def render_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    #pcd.paint_uniform_color([1, 0.706, 0])
    pcd.paint_uniform_color([26.0 / 255.0, 76.0/255.0, 102.0/255.0])
    #lookat(numpy.ndarray[float64[3, 1]]) – The
    #up(numpy.ndarray[float64[3, 1]]) – The
    #front(numpy.ndarray[float64[3, 1]]) – The
    #zoom(float)

    # front = np.asarray([[2], [2], [2]])
    #lookat = np.asarray([[0], [0], [0]], dtype=np.float64)

    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.8,
                                      front=[0.01, 1.0, 0.0],
                                      lookat=[0.0, 0.0, 0.0],
                                      up=[0.0, 1.0, 0.0])

file = "../local_datasets/toy9/data-2000/flange_1-b3.obj"

#### random sampled
points = create_point_cloud_from_file(file,
                                      n_points=2000,
                                      colors=False,
                                      n_min_points_surface_sampled=2000,
                                      all_mesh_vertices=False)
render_pcd(points)


#### poisson
points, normals = sample_poisson_from_mesh(file, n_points=2000, cache_dir=False)
render_pcd(points)

#### vertices
points = create_point_cloud_from_file(file, n_points=2000, colors=False, n_min_points_surface_sampled=0, all_mesh_vertices=True)
render_pcd(points)

#### vertices and nodes
points, normals = sample_nodes_and_poisson_from_mesh(file,
                                                     n_points_total=6000,
                                                     n_min_poisson=0,
                                                     cache_dir=False)
render_pcd(points)

#### poisson higher res
points, normals = sample_poisson_from_mesh(file, n_points=6000, cache_dir=False)
render_pcd(points)