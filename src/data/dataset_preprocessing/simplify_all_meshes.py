import open3d as o3d
from os import walk, path, remove, makedirs

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

if __name__ == '__main__':
    SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\shapenet\data'
    DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\shapenet\data-2000-open3d'
    target = 2000

    if not path.isdir(DEST_PATH):
        makedirs(DEST_PATH)

    max_edges = 0
    for (dirpath, dirnames, filenames) in walk(SRC_PATH):
        for filename in filenames:
            mesh = o3d.io.read_triangle_mesh(path.join(dirpath, filename))
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
            edge_count = count_edges(mesh)
            if edge_count > max_edges:
                max_edges = edge_count
            o3d.io.write_triangle_mesh(path.join(DEST_PATH, filename), mesh)
            #remove(path.join(DEST_PATH, f"{filename.split('.')[0]}.mtl"))

    print("max edge count:", max_edges) # sollte 4500 sein