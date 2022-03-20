import open3d as o3d
from os import walk, path, makedirs
from src.data.utils.manifold_simplification import simplify_mesh

# Tool: https://github.com/hjwdzh/Manifold
# Installed in ~/projects/Manifold

def run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT):
    if not path.isdir(DEST_PATH):
        makedirs(DEST_PATH)

    min_faces = 100000000000000
    max_faces = 0
    min_faces_before = 100000000000
    max_faces_before = 0

    for (dirpath, dirnames, filenames) in walk(SRC_PATH):
        for filename in filenames:
            if filename.split('.')[1] not in ['obj', 'stl']:
                continue

            print("filename", filename)
            src = path.join(dirpath, filename)

            if filename.split('.')[1] == 'stl':
                cache_dir = path.join(SRC_PATH, 'obj-cache')
                if not path.isdir(cache_dir):
                    makedirs(cache_dir)
                cached_file = path.join(cache_dir, filename.split('.')[0] + '.obj')
                if not path.isfile(cached_file):
                    mesh = o3d.io.read_triangle_mesh(src)
                    o3d.io.write_triangle_mesh(cached_file, mesh)
                src = cached_file

            # continue
            dest = path.join(DEST_PATH, f"{filename.split('.')[0]}.obj")

            # count faces before
            mesh = o3d.io.read_triangle_mesh(src)
            num_faces = len(mesh.triangles)
            if num_faces < min_faces_before:
                min_faces_before = num_faces
            if num_faces > max_faces_before:
                max_faces_before = num_faces

            # simplify
            simplify_mesh(src, dest, TARGET_FACE_COUNT)

            # count faces after
            mesh = o3d.io.read_triangle_mesh(dest)
            num_faces = len(mesh.triangles)
            if num_faces < min_faces:
                min_faces = num_faces
            if num_faces > max_faces:
                max_faces = num_faces

    print("min faces before:", min_faces_before)
    print("max faces before:", max_faces_before)
    print("min faces after:", min_faces)
    print("max faces after:", max_faces)

if __name__ == '__main__':
    SRC_PATH = '../../../datasets/eh_pro/data/'
    DEST_PATH = '../../../datasets/eh_pro/data-3000-watertight/'
    TARGET_FACE_COUNT = 3000
    run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
