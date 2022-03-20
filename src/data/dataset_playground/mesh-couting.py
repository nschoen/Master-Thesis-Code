import open3d as o3d

# def count_edges(mesh):
#     mesh.compute_adjacency_list()
#     edge_list = []
#     i = 0
#     for al in mesh.adjacency_list:
#         for r in al:
#             idx = f"{min(i, r)}-{max(i, r)}"
#             if idx not in edge_list:
#                 edge_list.append(idx)
#         i += 1
#     return len(edge_list)

def count_edges(mesh):
    mesh.compute_adjacency_list()
    edge_count = 0
    for i, al in enumerate(mesh.adjacency_list):
        for r in al:
            if i < r:
                edge_count += 1
    return edge_count


#mesh = o3d.io.read_triangle_mesh("../../../datasets/eh_pro/data-obj-folder-structured-simplified/housing/train/230045953-A.obj")
mesh = o3d.io.read_triangle_mesh("./230005494-d-10.stl")

print("#edges", count_edges(mesh))
print("#vertices", len(mesh.vertices))
print("Is Manofold?", mesh.is_edge_manifold())
print("Is Watertight?", mesh.is_watertight())

mesh.paint_uniform_color([1, 0.706, 0])
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

print("done")
