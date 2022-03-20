import open3d as o3d
from os import walk, path, remove, makedirs

def count_edges(mesh):
    mesh.compute_adjacency_list()
    edge_count = 0
    for i, al in enumerate(mesh.adjacency_list):
        for r in al:
            if i < r:
                edge_count += 1
    return edge_count


if __name__ == '__main__':
    SRC_PATH = '../../../datasets/eh_pro/data-obj/'
    DEST_PATH = '../../../datasets/eh_pro/data-obj-simplified/'

    makedirs(DEST_PATH, exist_ok=True)

    min_edges = 10000000000000
    max_edges = 0
    for (dirpath, dirnames, filenames) in walk(SRC_PATH):
        for filename in filenames:
            mesh = o3d.io.read_triangle_mesh(path.join(dirpath, filename))
            edge_count = count_edges(mesh)

            # if not mesh.is_edge_manifold():
            #     print("Manifold?", mesh.is_edge_manifold())
            # if not mesh.is_watertight():
            #     print("Watertight?", mesh.is_watertight())

            while edge_count < 20000:
                mesh = mesh.subdivide_midpoint(1)
                edge_count = count_edges(mesh)

            if edge_count > 50000:
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000) # 3000

            # if not mesh.is_edge_manifold():
            #     print("Manifold?", mesh.is_edge_manifold())
            # if not mesh.is_watertight():
            #     print("Watertight?", mesh.is_watertight())

            # mesh.remove_duplicated_triangles()
            mesh = mesh.remove_duplicated_vertices()
            mesh = mesh.remove_degenerate_triangles()
            mesh = mesh.remove_non_manifold_edges()  # important

            # get_non_manifold_vertices
            # get_non_manifold_edges
            # is_vertex_manifold

            # important, otherwise it can happen in mesh_prepare.slide_verts (post augmentation)
            # randomly selected vertices do not have edges and thus leading to an error (ValueError: min() arg is an empty sequence)
            mesh = mesh.remove_unreferenced_vertices()

            if not mesh.is_edge_manifold():
                print("Immernoch kein Manifold!", filename)

            edge_count = count_edges(mesh)

            if edge_count > max_edges:
                max_edges = edge_count
            if edge_count < min_edges:
                min_edges = edge_count
            o3d.io.write_triangle_mesh(path.join(DEST_PATH, filename), mesh)
            #remove(path.join(DEST_PATH, f"{filename.split('.')[0]}.mtl"))

    print("min edge count:", min_edges)
    print("max edge count:", max_edges) # sollte 4500 sein