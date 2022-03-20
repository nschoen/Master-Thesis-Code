import open3d as o3d

def count_edges(mesh):
    mesh.compute_adjacency_list()
    edge_list = []
    i = 0
    for al in mesh.adjacency_list:
        for r in al:
            idx = f"{min(i, r)}-{max(i, r)}"
            if idx not in edge_list:
                edge_list.append(idx)
        i += 1
    return len(edge_list)

mesh = o3d.io.read_triangle_mesh("./flange_21-b0.obj")
#mesh = o3d.io.read_triangle_mesh("./flange_849-b7.obj")
mesh.paint_uniform_color([1, 0.706, 0])
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
print("vertice size", mesh.vertices)

# print("0:", count_edges(mesh))
# mesh = mesh.subdivide_midpoint(1)
# print("1:", count_edges(mesh))
# # mesh = mesh.subdivide_midpoint(1)
# # print("2:", count_edges(mesh))
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])
print("triangles count", len(mesh.triangles))

# simpler
mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=3000) # 2500 looks good, good compromise, 5000 kaum unterscheidbar, edges count x1.5
mesh_smp.compute_vertex_normals()
edge_count = count_edges(mesh_smp)
print("edge_count", edge_count)
print("vertice count", len(mesh_smp.vertices))
o3d.visualization.draw_geometries([mesh_smp])

# more vertices
#mesh = mesh.subdivide_midpoint(2)
#mesh.compute_vertex_normals()
#o3d.visualization.draw_geometries([mesh])

print("done")