import open3d as o3d
from os import walk, path

max_edge_count = 0
max_edge_count_filename = None
vertice_count = 0

min_edge_count_filename = None
min_edge_count = 10000000

for (dirpath, dirnames, filenames) in walk('../datasets/eh_mmm_custom_flanges/data-obj-simplified/'):
    for filename in filenames:
        if filename.split('.')[1] != 'obj':
            continue
        mesh = o3d.io.read_triangle_mesh(path.join(dirpath, filename))
        #mesh.paint_uniform_color([1, 0.706, 0])
        mesh.compute_adjacency_list()
        #tri = mesh.triangles
        #ladiad = mesh.adjacency_list
        #edge_count = len(mesh.adjacency_list)

        edge_list = []
        i = 0
        for al in mesh.adjacency_list:
            for r in al:
                idx = f"{min(i, r)}-{max(i, r)}"
                if idx not in edge_list:
                    edge_list.append(idx)
            i += 1
        edge_count = len(edge_list)

        #if len(mesh.vertices) > max_edge_count:
        if edge_count > max_edge_count:
            max_edge_count = edge_count
            max_edge_count_filename = filename
            vertice_count = len(mesh.vertices)

        if edge_count < min_edge_count:
            min_edge_count = edge_count
            min_edge_count_filename = filename

print("max_edge_count_filename", max_edge_count_filename)
print("highest edge count", max_edge_count)
print("vertice_count", vertice_count)

print("min_edge_count_filename", min_edge_count_filename)
print("min_edge_count", min_edge_count)
