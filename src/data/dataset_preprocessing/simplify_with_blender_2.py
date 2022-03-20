import open3d as o3d
from os import walk, path, makedirs
from src.data.utils.blender_simplification import simplify_mesh
from tqdm import tqdm
from random import randrange

def run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT, range_max=None):
    if not path.isdir(DEST_PATH):
        makedirs(DEST_PATH)

    min_faces = 100000000000000
    max_faces = 0

    min_faces_before = 100000000000
    max_faces_before = 0

    for (dirpath, dirnames, filenames) in walk(SRC_PATH):
        #for filename in filenames:
        for filename in tqdm(filenames, total=len(filenames), smoothing=0.9):
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
            target_face_count = TARGET_FACE_COUNT
            if range_max:
                target_face_count = randrange(TARGET_FACE_COUNT, range_max)

            simplify_mesh(src, dest, target_face_count)

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
    return (min_faces_before, max_faces_before, min_faces, max_faces)

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

    # stats = []
    #
    # SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data'
    # DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data-2200'
    # TARGET_FACE_COUNT = 2200
    # min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    # stats.append(['2200', min_faces_before, max_faces_before, min_faces, max_faces])
    #
    # DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data-2400'
    # TARGET_FACE_COUNT = 2400
    # min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    # stats.append(['2400', min_faces_before, max_faces_before, min_faces, max_faces])
    #
    # DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data-2600'
    # TARGET_FACE_COUNT = 2600
    # min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    # stats.append(['2600', min_faces_before, max_faces_before, min_faces, max_faces])
    #
    # print(stats)

    stats = []
    #
    # for i in range(17):
    #     version = str(4 + i)
    #     SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\toy-final\data'
    #     DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\toy-final\dat a-1600-2100-aug' + version
    #     TARGET_FACE_COUNT = 1600
    #     RANGE_MAX = 2100
    #     min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT, range_max=RANGE_MAX)
    #     stats.append(['1600-2100-aug' + version, min_faces_before, max_faces_before, min_faces, max_faces])

    # stats = []
    #
    # for i in range(20):
    #     version = str(1 + i)
    #     SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data'
    #     DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data-1600-2100-aug' + version
    #     TARGET_FACE_COUNT = 1600
    #     RANGE_MAX = 2100
    #     min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT, range_max=RANGE_MAX)
    #     stats.append(['1600-2100-aug' + version, min_faces_before, max_faces_before, min_faces, max_faces])

    # SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\toy-final\data'
    # DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\toy-final\data-1700'
    # TARGET_FACE_COUNT = 1700
    # min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    # stats.append(['1700', min_faces_before, max_faces_before, min_faces, max_faces])

    # DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\toy-final\data-3400'
    # TARGET_FACE_COUNT = 3400
    # min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    # stats.append(['3400', min_faces_before, max_faces_before, min_faces, max_faces])
    #
    # DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\toy-final\data-1800'
    # TARGET_FACE_COUNT = 1800
    # min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    # stats.append(['1800', min_faces_before, max_faces_before, min_faces, max_faces])

    # SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data'
    # DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data-1600'
    # TARGET_FACE_COUNT = 1600
    # min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    # stats.append(['1600', min_faces_before, max_faces_before, min_faces, max_faces])
    #
    # SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data'
    # DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\data-1800'
    # TARGET_FACE_COUNT = 1800
    # min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    # stats.append(['1800', min_faces_before, max_faces_before, min_faces, max_faces])

    #SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_pro\data'
    #DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\eh_pro\data-1800'
    #TARGET_FACE_COUNT = 1800
    #min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    #stats.append(['1800', min_faces_before, max_faces_before, min_faces, max_faces])

    #SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\shapenet\data'
    #DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\shapenet\data-2000-no-triangulate'
    #TARGET_FACE_COUNT = 2000
    #min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    #stats.append(['2000', min_faces_before, max_faces_before, min_faces, max_faces])

    SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\toy3\data'
    DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\toy3\data-1600'
    TARGET_FACE_COUNT = 1600
    min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    stats.append(['1600', min_faces_before, max_faces_before, min_faces, max_faces])

    SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\toy3\data'
    DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\toy3\data-1800'
    TARGET_FACE_COUNT = 1800
    min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    stats.append(['1800', min_faces_before, max_faces_before, min_faces, max_faces])

    SRC_PATH = r'C:\Users\i00110578\projects\local_datasets\toy3\data'
    DEST_PATH = r'C:\Users\i00110578\projects\local_datasets\toy3\data-2000'
    TARGET_FACE_COUNT = 2000
    min_faces_before, max_faces_before, min_faces, max_faces = run(SRC_PATH, DEST_PATH, TARGET_FACE_COUNT)
    stats.append(['2000', min_faces_before, max_faces_before, min_faces, max_faces])

    print(stats)


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