from os import walk, path, makedirs
from src.data.utils.manifold_watertight import watetight_mesh

def run(SRC_PATH, DEST_PATH):
    if not path.isdir(DEST_PATH):
        makedirs(DEST_PATH)

    for (dirpath, dirnames, filenames) in walk(SRC_PATH):
        for filename in filenames:
            if filename.split('.')[1] not in ['obj']:
                continue

            print("filename", filename)
            src = path.join(dirpath, filename)
            dest = path.join(DEST_PATH, filename)
            watetight_mesh(src, dest)


if __name__ == '__main__':
    SRC_PATH = '../../../datasets/eh_pro/data/obj-cache'
    DEST_PATH = '../../../datasets/eh_pro/data-watertight-depth-8'
    run(SRC_PATH, DEST_PATH)
