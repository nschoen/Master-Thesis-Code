import open3d as o3d
import numpy as np
import os


def get_cached_pc(file, cache_dir):
    basename = os.path.basename(file).split('.')[0]
    cache_file = os.path.join(cache_dir, f"{basename}.ply")
    if cache_dir and os.path.isfile(cache_file):
        pcd = o3d.io.read_point_cloud(cache_file, format='auto', print_progress=False)
        return np.asarray(pcd.points), np.asarray(pcd.normals)
    return False


def cache_pc(pc, file, cache_dir):
    basename = os.path.basename(file).split('.')[0]
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"{basename}.ply")
        o3d.io.write_point_cloud(cache_file, pc)


def sample_poisson_from_mesh(file, n_points=1000, cache_dir=None):
    if cache_dir:
        cached_file = get_cached_pc(file, cache_dir)
        if cached_file:
            return cached_file

    mesh = o3d.io.read_triangle_mesh(file)
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    try:
        pc = mesh.sample_points_poisson_disk(n_points)
    except:
        print("file", file)
    pc = mesh.sample_points_poisson_disk(n_points)

    if cache_dir:
        cache_pc(pc, file, cache_dir)

    return np.asarray(pc.points), np.asarray(pc.normals)


def sample_nodes_and_poisson_from_mesh(file,
                                       n_points_total=1000,
                                       n_min_poisson=0,
                                       cache_dir=False):
    """Support stl and off files"""
    assert file.split('.')[-1] in ['stl', 'off', 'obj']

    if cache_dir:
        cached_file = get_cached_pc(file, cache_dir)
        if cached_file:
            return cached_file

    mesh = o3d.io.read_triangle_mesh(file)
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    pc = o3d.geometry.PointCloud(points=mesh.vertices)
    pc.normals = mesh.vertex_normals

    n_node_points = np.asarray(pc.points).shape[0]
    n_poisson = max(n_points_total - n_node_points, n_min_poisson)
    n_max_node_points = n_points_total - n_min_poisson

    if n_max_node_points < n_node_points:
        pc.points = o3d.utility.Vector3dVector(np.asarray(pc.points)[np.random.choice(n_node_points, n_max_node_points, replace=False)])
        pc.normals = o3d.utility.Vector3dVector(np.asarray(pc.normals)[np.random.choice(n_node_points, n_max_node_points, replace=False)])

    if n_poisson > 0:
        pc_poisson = mesh.sample_points_poisson_disk(n_poisson)
        pc.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pc.points), np.asarray(pc_poisson.points))))
        pc.normals = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pc.normals), np.asarray(pc_poisson.normals))))

    if cache_dir:
        cache_pc(pc, file, cache_dir)

    #return all_vertices
    return np.asarray(pc.points), np.asarray(pc.normals)





##################
##################
#### Rest is irrelevant
####

# STILL USED for example comparison of random vs poissson
def create_point_cloud_from_file(file, n_points=15000, colors=False, n_min_points_surface_sampled=200, all_mesh_vertices=False):
    """Support stl and off files"""
    assert file.split('.')[-1] in ['stl', 'off', 'obj']

    # read stl file
    mesh = o3d.io.read_triangle_mesh(file)

    # get vertices and triangles
    # the vertices are used as the initial points
    # the triangles are used to sample additional points uniformly according to their sizes
    vertices = np.asarray(mesh.vertices)
    if all_mesh_vertices:  # and n_min_points_surface_sampled == 0:
        return vertices

    triangles = np.asarray(mesh.triangles)

    # if len(triangles) == 0:
    #     print("0 triangles found for file", file)
    #     return None

    n_initial_points = vertices.shape[0]
    # print("vertices.shape", vertices.shape)
    t_xyz = np.apply_along_axis(put_vertices_into_triangles, 1, triangles, vertices)
    # except ValueError:
    #    print("Unexpected error:", sys.exc_info()[0])
    #    print(triangles)
    #    raise

    # print(t_xyz)
    # n_initial_points = t_xyz.shape[0]
    # print("n_points", n_initial_points)

    # https://stackoverflow.com/questions/45973722/how-does-numpy-reshape-with-order-f-work
    # v_xyz = t_xyz.ravel('F').reshape((3, t_xyz.shape[0], 3), order='F')

    v_1 = t_xyz[:, 0, :]
    v_2 = t_xyz[:, 1, :]
    v_3 = t_xyz[:, 2, :]

    # print(v_1)
    # print(v_2)
    # print(v_3)

    # calculate the size for each triangle
    areas = triangle_area_multi(v_1, v_2, v_3)
    probabilities = areas / areas.sum()
    # print("areas.size", areas.size)

    # use the size of each triangle as its probability to contain the next point
    # and use it to sample n - n_initial_points
    n_rest = max(n_points - n_initial_points, n_min_points_surface_sampled)
    n_max_original = n_points - n_min_points_surface_sampled
    original_vertices = vertices

    if n_max_original < n_initial_points:
        # want an equal number of points for each point cloud, reduce initial vertices if there are too many
        # print(os.path.basename(file), "has to many initial vertices", n_initial_points, "but max is", n_max_original, "=> sample random n vertices")
        # print(np.random.choice(vertices.shape[0], n, replace=False))
        all_but_n_vertices = vertices[np.random.choice(vertices.shape[0], n_max_original, replace=False)]
        original_vertices = all_but_n_vertices

    if n_rest > 0:
        weighted_random_indices = np.random.choice(range(len(areas)), size=n_rest, p=probabilities)
        weighted_random_indices

        v1_selected = v_1[weighted_random_indices]
        v2_selected = v_2[weighted_random_indices]
        v3_selected = v_3[weighted_random_indices]

        # now within the selected triangles the points have to be uniformly distributed on the given area
        u = np.random.rand(n_rest, 1)
        v = np.random.rand(n_rest, 1)
        is_a_problem = u + v > 1
        u[is_a_problem] = 1 - u[is_a_problem]
        v[is_a_problem] = 1 - v[is_a_problem]
        w = 1 - (u + v)

        new_points = (v1_selected * u) + (v2_selected * v) + (v3_selected * w)
        new_points = new_points.astype(np.float32)
        # print("new points shape", new_points.shape)
        # print("vertices shape", np.asarray(mesh.vertices).shape)
        all_vertices = np.append(original_vertices, new_points, axis=0)
    else:
        all_vertices = original_vertices

    # print("all vertice shape", all_vertices.shape)

    return all_vertices

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(all_vertices)
    #
    # if colors is False:
    #     pcd.paint_uniform_color([0.5, 0.5, 0.5])
    #
    # return pcd


def convert_to_point_cloud_from_directory(source_path, target_path, n=15000, n_min_triangled_samples=200):
    for (dirpath, dirnames, filenames) in os.walk(source_path):
        for filename in filenames:
            basename = filename.split('.')[0]
            # relpath = os.path.relpath(dirpath, source_path)

            file = os.path.join(dirpath, filename)
            pcd = create_point_cloud_from_file(file, n=n, n_min_triangled_samples=n_min_triangled_samples)

            if pcd == None:
                continue

            target_file = os.path.join(target_path, f"{basename}.pcd")
            # target_file = os.path.join(target_path, relpath, f"{basename}.pcd")
            # os.makedirs(os.path.join(target_path, relpath), exist_ok=True)
            o3d.io.write_point_cloud(target_file, pcd)


def read_pcd(file):
    return o3d.io.read_point_cloud(file)


def read_pcd_dir(path):
    pcds = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            pcds.append({
                "pcd": read_pcd(os.path.join(dirpath, filename)),
                "name": filename.split('.')[0],
            })

    return pcds


def triangle_area_multi(v1, v2, v3):
    """Compute the are of multiple triangles in face-vertex format
    Parameters
    ---------
    """
    # print(v2-v1)
    return 0.5 * np.linalg.norm(np.cross(v2 - v1,
                                         v3 - v1), axis=1)


def put_vertices_into_triangles(t, v):
    """Replaces the vertice indices in the triangle arrays by the actual vertices"""
    return [v[t[0]], v[t[1]], v[t[2]]]