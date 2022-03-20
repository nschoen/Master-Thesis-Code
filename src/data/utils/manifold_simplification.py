import open3d as o3d
import os
import subprocess

MANIFOLD_SIMPLIFICATION_SCRIPT = "/home/i00110578/projects/Manifold/build/simplify"
TEMP_DIR = './tmp'

if not os.path.isdir(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def simplify_mesh(src, dest, target_faces=1000):
    if os.path.isfile(dest):
        return
    mesh = o3d.io.read_triangle_mesh(src)
    #mesh = mesh.merge_close_vertices(0.001)
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_degenerate_triangles()
    new_src = os.path.join(TEMP_DIR, os.path.basename(src))
    o3d.io.write_triangle_mesh(new_src, mesh)
    print("asd", new_src)

    subprocess.call([MANIFOLD_SIMPLIFICATION_SCRIPT, '-i', new_src, '-o', dest, '-m', '-f', str(target_faces)])

    # mesh = o3d.io.read_triangle_mesh(dest)
    # print("Manifold?", mesh.is_edge_manifold())
    # mesh = mesh.merge_close_vertices(0.0001)
    # mesh = mesh.remove_duplicated_vertices()
    # mesh = mesh.remove_degenerate_triangles()
    # mesh = mesh.remove_non_manifold_edges()
    # print("Watertight?", mesh.is_watertight())
    # o3d.io.write_triangle_mesh(dest, mesh)