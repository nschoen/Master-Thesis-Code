import open3d as o3d
from os import walk
from os.path import join

def count_edges(mesh):
    mesh.compute_adjacency_list()
    edge_count = 0
    for i, al in enumerate(mesh.adjacency_list):
        for r in al:
            if i < r:
                edge_count += 1
    return edge_count


min_edge = 100000000000000
max_edge = 0
min_vertices = 100000000000000
max_vertices = 0
min_triangles = 100000000000000
max_triangles = 0
filename_min = None
filename_max = None
empty_meshes = []
num_empty_meshes = 0

files_lower_1500 = []

extension = '.obj' # .obj / .stl

for (dirpath, dirnames, filenames) in walk("../../../datasets/mcb_b/data"): # stl-default
    for filename in filenames:
        if extension not in filename:
            continue
        mesh = o3d.io.read_triangle_mesh(join(dirpath, filename))
        edge_count = count_edges(mesh)
        vertice_count = len(mesh.vertices)
        triangle_count = len(mesh.triangles)
        if vertice_count == 0:
            empty_meshes.append(filename)
            num_empty_meshes += 1
            continue
        if edge_count < min_edge:
            min_edge = edge_count
        if edge_count > max_edge:
            max_edge = edge_count
        # if edge_count < 1500:
        #     files_lower_1500.append({ 'number': edge_count, "filename": filename})
        if vertice_count < min_vertices:
            min_vertices = vertice_count
        if vertice_count > max_vertices:
            max_vertices = vertice_count
        if triangle_count < min_triangles:
            min_triangles = triangle_count
            filename_min = filename
        if triangle_count > max_triangles:
            max_triangles = triangle_count
            filename_max = filename

print("min edges", min_edge)
print("max edges", max_edge)
print("min vertices", min_vertices)
print("max vertices", max_vertices)
print("min triangles", min_triangles)
print("max triangles", max_triangles)
print("filename_min", filename_min)
print("filename_max", filename_max)
print("empty_meshes", empty_meshes)
print("num_empty_meshes", num_empty_meshes)

# print("files_lower_1500", files_lower_1500)
# min is 0??? fo which?
# => 4730 for now max_edges
# min is 1536

# bei collapse zu 1500 faces
# min edges 1536
# max edges 2346


#files_lower_500 [{'number': 0, 'filename': '230046668-G.obj'}, {'number': 0, 'filename': '230046662-L.obj'}]


# Results:
#
# data-creo-default
# min edges 768
# max edges 8730
# min vertices 768
# max vertices 8730
# min triangles 256
# max triangles 2910
# filename_min flange_17-b0.stl
# filename_max flange_14-b7.stl
#
# data-creo-stepsize
# min edges 7074
# max edges 16500
# min vertices 7074
# max vertices 16500
# min triangles 2358
# max triangles 5500
# filename_min flange_20-b0.stl
# filename_max flange_14-b7.stl
#
# data-creo-stepsize-2000
# min edges 3942
# max edges 5922
# min vertices 2355
# max vertices 5806
# min triangles 1998
# max triangles 2000
# filename_min flange_2-b5.obj
# filename_max flange_34-b8.obj
#
# data-freecad
#
#
# data-freecad-2000:
# min edges 3946
# max edges 5776
# min vertices 2219
# max vertices 5555
# min triangles 1998
# max triangles 2000
# filename_min flange_2-b5.obj
# filename_max flange_849-x1-4.obj
#

# EH PRO
#
# Feedcad default
# min edges 3072
# max edges 430932
# min vertices 3072
# max vertices 430932
# min triangles 1024
# max triangles 143644
# filename_min 230005751-A.stl
# filename_max 230034691-A.stl
#
# FreeCad 600
# min edges 661
# max edges 1964
# min vertices 230
# max vertices 599
# min triangles 391
# max triangles 1237
# filename_min 231037144-A.obj
# filename_max 230043160-D.obj
#
# FreeCad 2000
# min edges 2466
# max edges 3135
# min vertices 759
# max vertices 1120
# min triangles 1557
# max triangles 2000
# filename_min 230043160-D.obj
# filename_max 230007699-A.obj
#
# FreeCad 4000
# min edges 5514
# max edges 6240
# min vertices 1811
# max vertices 2208
# min triangles 3556
# max triangles 4000
# filename_min 230043160-D.obj
# filename_max 230037451--.obj
#

# mcb_a
#
# data
# min edges 6
# max edges 5272701
# min vertices 4
# max vertices 1747648
# min triangles 4
# max triangles 3539799
# filename_min 00048649.obj
# filename_max 00031554.obj
# empty_meshes ['00010798.obj']
# num_empty_meshes 1
# with 00010798.obj removed
#

# mcb_b
#
# min edges 6
# max edges 5272701
# min vertices 4
# max vertices 1747648
# min triangles 4
# max triangles 3539799
# filename_min 00000077.obj
# filename_max 00071022.obj
# empty_meshes []
# num_empty_meshes 0


