import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("../local_datasets/toy9/data-2000/flange_1-b3.obj")

print("Is Manofold?", mesh.is_edge_manifold())
print("Is Watertight?", mesh.is_watertight())

mesh.paint_uniform_color([1, 0.706, 0])
#mesh.paint_uniform_color([26.0 / 255.0, 76.0/255.0, 102.0/255.0])
mesh = mesh.compute_vertex_normals()
#o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)


triangles = np.asarray(mesh.triangles)
triangle_normals = np.asarray(mesh.triangle_normals)
vertices = np.asarray(mesh.vertices)

face_normals = {}
edge_faces = {}
edge_normals = {}
edge_pos = {}
i = 0

for t in triangles:
    v0 = vertices[t[0]]
    v1 = vertices[t[1]]
    v2 = vertices[t[2]]

    id0 = '-'.join([str(v0[0]), str(v0[1]), str(v0[2])])
    id1 = '-'.join([str(v1[0]), str(v1[1]), str(v1[2])])
    id2 = '-'.join([str(v2[0]), str(v2[1]), str(v2[2])])
    pos_map = {}
    pos_map[id0] = v0
    pos_map[id1] = v1
    pos_map[id2] = v2

    face_id = '-'.join(sorted([id0, id1, id2]))
    face_normals[face_id] = triangle_normals[i]

    for id_a, id_b in [[id0, id1], [id1, id2], [id0, id2]]:
        edge_id = '-'.join(sorted([id_a, id_b]))
        if not edge_id in edge_faces:
            edge_faces[edge_id] = []
        edge_faces[edge_id].append(face_id)

        if len(edge_faces[edge_id]) == 2:
            # both faces provided, calculate edge normal
            normal1 = face_normals[edge_faces[edge_id][0]]
            normal2 = face_normals[edge_faces[edge_id][1]]
            edge_normals[edge_id] = (normal1 + normal2) / 2
            edge_pos[edge_id] = (pos_map[id_a] + pos_map[id_b]) / 2  # correct one
            # edge_pos[edge_id] = pos_map[id_a] - (pos_map[id_a] - pos_map[id_b]) / 2  # works as well...

    i += 1

final_vertices = []
final_normals = []

for edge_id, pos in edge_pos.items():
    final_vertices.append(pos)
    final_normals.append(edge_normals[edge_id])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(final_vertices))
pcd.normals = o3d.utility.Vector3dVector(np.array(final_normals))
pcd.paint_uniform_color([26.0 / 255.0, 76.0/255.0, 102.0/255.0])
#pcd.estimate_normals()
o3d.visualization.draw_geometries([mesh, pcd], mesh_show_wireframe=True)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
# pcd.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
# #pcd.estimate_normals()
# o3d.visualization.draw_geometries([pcd])


#pcd = mesh.sample_points_poisson_disk(2000)