import open3d as o3d

mesh = o3d.io.read_triangle_mesh('../../../datasets/eh_pro/data-watertight-depth-8/230007699-A.obj')
print("Manifold?", mesh.is_edge_manifold())
# mesh = mesh.merge_close_vertices(0.0001)
# mesh = mesh.remove_duplicated_vertices()
# mesh = mesh.remove_degenerate_triangles()
# mesh = mesh.remove_non_manifold_edges()
print("Watertight?", mesh.is_watertight())
# o3d.io.write_triangle_mesh(dest, mesh)