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

#mesh = o3d.io.read_triangle_mesh("../../../datasets/eh_pro/data-obj/230005809-G.obj")
mesh = o3d.io.read_triangle_mesh("../../../datasets/eh_pro/data-obj/230006883-C.obj")

print("#edges", count_edges(mesh))
print("#vertices", len(mesh.vertices))
print("Is Manofold?", mesh.is_edge_manifold())
print("Is Watertight?", mesh.is_watertight())

mesh.paint_uniform_color([1, 0.706, 0])
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

#mesh = mesh.subdivide_loop()
# mesh = mesh.subdivide_midpoint() => prefered, looks better

mesh = mesh.subdivide_midpoint(2)

# mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=1500)
# mesh = mesh.remove_duplicated_vertices()
# mesh = mesh.remove_degenerate_triangles()
# mesh = mesh.remove_non_manifold_edges()
# mesh = mesh.remove_degenerate_triangles()
# mesh = mesh.merge_close_vertices(0.1)

print("#edges", count_edges(mesh))
print("#vertices", len(mesh.vertices))
print("Is Manofold?", mesh.is_edge_manifold())
print("Is Watertight?", mesh.is_watertight())

mesh.paint_uniform_color([1, 0.706, 0])
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

print("done")
