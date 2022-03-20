import open3d as o3d

if __name__ == '__main__':
    mesh_path = r'C:\Users\i00110578\projects\local_datasets\toy-final\data-1600\flange_849-b0.obj'
    # flange_2-b7.stl b2, b3, b7
    # flange_357-b4.stl  b7, b0, b1
    # flange_849 - b7.stl b0
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    pc = mesh.sample_points_poisson_disk(3000)
    #pc.points = pc.normals
    pc.points = mesh.vertex_normals
    pc.paint_uniform_color([0.3, 0.3, 0.3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pc, mesh_frame])
