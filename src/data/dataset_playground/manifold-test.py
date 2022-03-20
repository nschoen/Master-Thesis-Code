import open3d as o3d
from os import walk, path

meshes_with_non_manifolds = []
meshes_with_non_manifolds_counts = []

for (dirpath, dirnames, filenames) in walk('../datasets/eh_mmm_custom_flanges/data-obj-simplified/'):
    for filename in filenames:
        #print("filename.split('.')[1]", filename.split('.')[1])
        if filename.split('.')[1] != 'obj':
            continue
        mesh = o3d.io.read_triangle_mesh(path.join(dirpath, filename))
        #mesh.compute_adjacency_list()

        non_manifold_edges = mesh.get_non_manifold_edges()
        n_non_manifold_edges = len(non_manifold_edges)

        if n_non_manifold_edges > 0:
            meshes_with_non_manifolds.append(filename)
            meshes_with_non_manifolds_counts.append(non_manifold_edges)

print(meshes_with_non_manifolds)
print(meshes_with_non_manifolds_counts)
