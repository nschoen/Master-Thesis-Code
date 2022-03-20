import numpy as np
import itertools


def extract_edge_features(mesh):
    mesh = mesh.merge_close_vertices(0.000000000001)
    mesh.compute_adjacency_list()
    adjacency_list = mesh.adjacency_list
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    features = []
    edge_points = get_edge_points(adjacency_list, faces)

    with np.errstate(divide='raise'):
        try:
            for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios, set_edge_lengths, vertex_normal_cosine]:
                feature = extractor(vertices, edge_points)
                features.append(feature)
            return np.concatenate(features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad features')


def vertex_normal_cosine(vertices, edge_points):
    x_i = vertices[edge_points[:, 0]]
    x_j = vertices[edge_points[:, 1]]

    # bla = np.linalg.norm(x_i, axis=1)
    # print("np.linalg.norm(x_i, axis=1)", np.linalg.norm(x_i, axis=1))

    dot = x_i[:, 0] * x_j[:, 0] + x_i[:, 1] * x_j[:, 1] + x_i[:, 2] * x_j[:, 2]
    cos = dot / (np.linalg.norm(x_i, axis=1) * np.linalg.norm(x_j, axis=1))

    return np.expand_dims(cos, axis=0)


def dihedral_angle(vertices, edge_points):
    normals_a = get_normals(vertices, edge_points, 0)
    normals_b = get_normals(vertices, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles


def symmetric_opposite_angles(vertices, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(vertices, edge_points, 0)
    angles_b = get_opposite_angles(vertices, edge_points, 3)
    angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
    angles = np.sort(angles, axis=0)
    return angles


def symmetric_ratios(vertices, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(vertices, edge_points, 0)
    ratios_b = get_ratios(vertices, edge_points, 3)
    ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
    return np.sort(ratios, axis=0)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def get_edge_points(adjacency_list, faces):
    """ returns: edge_points (#E x 4) tensor, with four vertex ids per edge
        for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices which define edge_id
        each adjacent face to edge_id has another vertex, which is edge_points[edge_id, 2] or edge_points[edge_id, 3]
    """
    # edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    # for edge_id, edge in enumerate(mesh.edges):
    #     edge_points[edge_id] = get_side_points(mesh, edge_id)
    #     # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    #
    # [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]

    edges = []
    i = 0
    for al in adjacency_list:
        for r in al:
            # if i < r:
            edges.append([i, r])
        i += 1

    faces = [sorted(face) for face in faces]

    ring_edge_ids = []
    for edge_id, edge in enumerate(edges):
        adj_vertices_a = adjacency_list[edge[0]]
        adj_vertices_b = adjacency_list[edge[1]]

        # there must be two overlapping vertices in adj_vertices_a and adj_vertices_b
        intersect_vertices = intersection(adj_vertices_a, adj_vertices_b)

        #if edge_id == 286:
        #    print("asdsa")

        # make sure that the overlapping vertices and the vertices of the currenct edge actually create a face
        # because the efges vertices can have more than 2 common vertices but these do not create a face then
        # (imagine a pyramid with a diagonal in the base)
        intersect_vertices_face_filtered = []
        for v in intersect_vertices:
            if sorted([edge[0], edge[1], v]) in faces:
                intersect_vertices_face_filtered.append(v)
        neighbor_edge_vertices = list(itertools.product(intersect_vertices_face_filtered, edge))


        # now fine the index of the edges in the list
        ring_ids = []
        for edge_ver_pairs in neighbor_edge_vertices:
            edge_ver_pairs = sorted(list(edge_ver_pairs))
            ring_ids.append(edges.index(edge_ver_pairs))

        ring_edge_ids.append(ring_ids)

    ring_edge_ids = np.asarray(ring_edge_ids)

    edge_points = np.zeros([len(edges), 4], dtype=np.int32)
    for edge_id, edge in enumerate(edges):
        edge_points[edge_id] = get_side_points(edges, ring_edge_ids, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)

    return np.asarray(edge_points)


def get_side_points(edges, ring_edge_ids, edge_id):
    # if mesh.gemm_edges[edge_id, side] == -1:
    #     return mesh.get_side_points(edge_id, ((side + 2) % 4))
    # else:
    edge_a = edges[edge_id]

    ring_edge_ids.shape

    if ring_edge_ids[edge_id, 0] == -1:
        edge_b = edges[ring_edge_ids[edge_id, 2]]
        edge_c = edges[ring_edge_ids[edge_id, 3]]
    else:
        edge_b = edges[ring_edge_ids[edge_id, 0]]
        edge_c = edges[ring_edge_ids[edge_id, 1]]
    if ring_edge_ids[edge_id, 2] == -1:
        edge_d = edges[ring_edge_ids[edge_id, 0]]
        edge_e = edges[ring_edge_ids[edge_id, 1]]
    else:
        edge_d = edges[ring_edge_ids[edge_id, 2]]
        edge_e = edges[ring_edge_ids[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]


def set_edge_lengths(vertices, edge_points=None):
    # if edge_points is not None:
    #     edge_points = get_edge_points(mesh)
    edge_lengths = np.linalg.norm(vertices[edge_points[:, 0]] - vertices[edge_points[:, 1]], ord=2, axis=1)
    # mesh.edge_lengths = edge_lengths
    return np.expand_dims(edge_lengths, axis=0)


def get_normals(vertices, edge_points, side):
    edge_a = vertices[edge_points[:, side // 2 + 2]] - vertices[edge_points[:, side // 2]]
    edge_b = vertices[edge_points[:, 1 - side // 2]] - vertices[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals


def get_opposite_angles(vertices, edge_points, side):
    edges_a = vertices[edge_points[:, side // 2]] - vertices[edge_points[:, side // 2 + 2]]
    edges_b = vertices[edge_points[:, 1 - side // 2]] - vertices[edge_points[:, side // 2 + 2]]

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def get_ratios(vertices, edge_points, side):
    edges_lengths = np.linalg.norm(vertices[edge_points[:, side // 2]] - vertices[edge_points[:, 1 - side // 2]],
                                   ord=2, axis=1)
    point_o = vertices[edge_points[:, side // 2 + 2]]
    point_a = vertices[edge_points[:, side // 2]]
    point_b = vertices[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths


def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div