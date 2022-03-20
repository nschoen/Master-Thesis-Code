import open3d as o3d


#file = "../local_datasets/toy9/stl-freecad/flange_451-b6.obj"
#file = r"C:\Users\i00110578\projects\AIAx-Use-Case-1\datasets\MMM\data_obj\housing_133.obj"
#file = "../local_datasets/toy9/data-2000/flange_1-b3.obj"
file = "../local_datasets/toy9/stl-freecad/flange_2-b6-2.stl"
color = [100.0 / 255.0, 150.0/255.0, 102.0/255.0]
#color = [26.0 / 255.0, 76.0/255.0, 102.0/255.0]

def draw_geometry_flange(model, mesh_show_wireframe=False):
    o3d.visualization.draw_geometries([model],
                                      mesh_show_wireframe=mesh_show_wireframe,
                                      zoom=0.8,
                                      front=[0.93, 1.0, 0.4],
                                      lookat=[0.0, 0.0, 0.0],
                                      up=[0.0, 1.0, 0.0])

def draw_geometry(model, mesh_show_wireframe=False):
    o3d.visualization.draw_geometries([model],
                                      mesh_show_wireframe=mesh_show_wireframe,
                                      zoom=0.8,
                                      front=[-0.8, 1.0, 0.0],
                                      lookat=[0.0, 0.0, 0.0],
                                      up=[1.0, 0.0, 0.0])


#### mesh
mesh = o3d.io.read_triangle_mesh(file)
mesh.paint_uniform_color(color)
mesh.compute_vertex_normals()
draw_geometry(mesh, mesh_show_wireframe=True)

#### point cloud
pcd = mesh.sample_points_poisson_disk(9000)
pcd.paint_uniform_color(color)
draw_geometry(pcd)

#### voxel grid
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=3.5)
#voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=5.5)
voxels = voxel_grid.get_voxels()
vox_mesh = o3d.geometry.TriangleMesh()
for vox in voxels:
    cube = o3d.geometry.TriangleMesh.create_box()
    #cube.scale(0.005, center=cube.get_center())
    cube.translate(
        (
            vox.grid_index[0],
            vox.grid_index[1],
            vox.grid_index[2],
        ),
        relative=False,
    )
    vox_mesh += cube
vox_mesh.paint_uniform_color(color)
vox_mesh.compute_vertex_normals()
draw_geometry(vox_mesh)

#### images
o3d.visualization.draw_geometries([mesh],
                                  zoom=0.8,
                                  front=[0.85, 1.0, 1.0],
                                  lookat=[0.0, 0.0, 0.0],
                                  up=[0.0, 1.0, 0.0])