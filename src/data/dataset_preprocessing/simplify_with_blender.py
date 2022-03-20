import open3d as o3d
from os import walk, path, makedirs
from src.data.utils.blender_simplification import simplify_mesh

def run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT):
    if not path.isdir(DEST_PATH):
        makedirs(DEST_PATH)

    min_faces = 100000000000000
    max_faces = 0

    min_faces_before = 100000000000
    max_faces_before = 0

    for (dirpath, dirnames, filenames) in walk(SRC_PATH):
        for filename in filenames:
            if filename.split('.')[1] not in ['stl', 'obj']:
                continue

            print("filename", filename)
            src = path.join(dirpath, filename)
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
    # SRC_PATH = '../local_datasets/toy9/stl-freecad/'
    # DEST_PATH = '../local_datasets/toy9/data-2000/'
    # TARGET_FACE_COUNT = 2000
    # run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    #
    # SRC_PATH = '../local_datasets/toy9/stl-freecad/'
    # DEST_PATH = '../local_datasets/toy9/data-2400/'
    # TARGET_FACE_COUNT = 2400
    # run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    #
    # SRC_PATH = '../local_datasets/toy9/stl-freecad/'
    # DEST_PATH = '../local_datasets/toy9/data-2600/'
    # TARGET_FACE_COUNT = 2600
    # run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)

    #SRC_PATH = '../../../../local_datasets/mcb_b/data/'
    #DEST_PATH = '../../../../local_datasets/mcb_b/data-2048/'

    #SRC_PATH = '../local_datasets/simplification-test/'
    #DEST_PATH = '../local_datasets/simplification-test-target'
    #TARGET_FACE_COUNT = 2000
    #run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)

    #SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\toy-final\data'
    #DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\toy-final\data-2600'

    #SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data'
    #DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data-3000'

    SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\shapenet\data'
    DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\shapenet\data-2000'

    TARGET_FACE_COUNT = 2000
    run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)


# ../local_datasets/toy9/data-obj-creo-stepsize-2000/
# min faces before: 2358
# max faces before: 5500
# min faces after: 1998
# max faces after: 2000

# '../local_datasets/eh_pro/data-freecad-600/'


# '../local_datasets/eh_pro/data-freecad-1000/'


# '../local_datasets/eh_pro/data-freecad-2000/'
# min faces before: 1024
# max faces before: 143644
# min faces after: 1557
# max faces after: 2000

# '../local_datasets/eh_pro/data-freecad-4000/'
# min faces before: 1024
# max faces before: 143644
# min faces after: 3556
# max faces after: 4000

# '../local_datasets/eh_pro/data-freecad-10000/'
# min faces before: 1024
# max faces before: 143644
# min faces after: 6144
# max faces after: 10000