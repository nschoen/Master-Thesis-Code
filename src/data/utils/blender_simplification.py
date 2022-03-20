import open3d as o3d
import os
import subprocess

BLENDER_SCRIPT = r"C:\Program Files\Blender Foundation\Blender 2.83\blender.exe"
BLENDER_PROCESS = r"C:\Users\i00110578\projects\pointnet-eh-data\src\data\utils\blender_process.py"

if not os.path.isdir('./tmp'):
    os.makedirs('./tmp')

def simplify_mesh(src, dest, target_faces=1000):
    if os.path.isfile(dest):
        return
    mesh = o3d.io.read_triangle_mesh(src)
    basename = os.path.basename(src).replace('.stl', '.obj')
    src = os.path.abspath(os.path.join('./tmp', basename))
    mesh = mesh.merge_close_vertices(0.0001)
    #mesh = mesh.remove_duplicated_vertices()
    #mesh = mesh.remove_degenerate_triangles()
    #mesh = mesh.remove_non_manifold_edges()
    o3d.io.write_triangle_mesh(src, mesh)

    subprocess.call([BLENDER_SCRIPT, '--background', '--python', BLENDER_PROCESS, src, str(target_faces), dest])
    #os.system(f"'{BLENDER_SCRIPT}' --background --python '{BLENDER_PROCESS}' '{src}' {target_faces} '{dest}'")

    mesh = o3d.io.read_triangle_mesh(dest)
    mesh = mesh.merge_close_vertices(0.0001)
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_non_manifold_edges()
    o3d.io.write_triangle_mesh(dest, mesh)
