from os.path import join, isfile
import csv
import json
from shutil import copyfile

DEST_DIR = '../../../datasets/shapenetcore_partanno_segmentation_benchmark_v0'
SHAPENET_DIR = "../../../../shapenet/shapenetcore_partanno_segmentation_benchmark_v0_normal"
SHAPENET_MESH_DIR = "../../../../shapenet/ShapeNetCore.v2"

class_ids = {
    "02691156": "Airplane",
    "02773838": "Bag",
    "02954340": "Cap",
    "02958343": "Car",
    "03001627": "Chair",
    "03261776": "Earphone",
    "03467517": "Guitar",
    "03624134": "Knife",
    "03636649": "Lamp",
    "03642806": "Laptop",
    "03790512": "Motorbike",
    "03797390": "Mug",
    "03948459": "Pistol",
    "04099429": "Rocket",
    "04225987": "Skateboard",
    "04379243": "Table",
}

class_ids_id = {
    "02691156": 0,
    "02773838": 1,
    "02954340": 2,
    "02958343": 3,
    "03001627": 4,
    "03261776": 5,
    "03467517": 6,
    "03624134": 7,
    "03636649": 8,
    "03642806": 9,
    "03790512": 10,
    "03797390": 11,
    "03948459": 12,
    "04099429": 13,
    "04225987": 14,
    "04379243": 15,
}

class_name_to_id = {
    "Airplane": "02691156",
    "Bag": "02773838",
    "Cap": "02954340",
    "Car": "02958343",
    "Chair": "03001627",
    "Earphone": "03261776",
    "Guitar": "03467517",
    "Knife": "03624134",
    "Lamp": "03636649",
    "Laptop": "03642806",
    "Motorbike": "03790512",
    "Mug": "03797390",
    "Pistol": "03948459",
    "Rocket": "04099429",
    "Skateboard": "04225987",
    "Table": "04379243",
}

# ignore these, are not present in ShapeNetCore V2
ignore_filenames = [
    "6d619fcbceaa0327104b57ee967e352c",
    "92d0fa7147696cf5ba531e418cb6cd7d",
    "f59a474f2ec175eb7cdba8f50ac8d46c",
    "5b04b836924fe955dab8f5f5224d1d8a",
    "4253a9aac998848f664839bbd828e448",
    "5973afc979049405f63ee8a34069b7c5",
    "4ddef66f32e1902d3448fdcb67fe08ff",
    "a8dde04ca72c5bdd6ca2b6e5474aad11",
    "8b68f086176443b8128fe65339f3ddb2",
    "c7bf88ef123ed4221694f51f0d69b70d",
    "8843d862a7545d0d96db382b382d7132",
    "a18fd5cb2a9d01c4158fe40320a23c2",
    "29a4e6ae1f9cecab52470de2774d6099",
    "a81cb450ce415d45bdb32c3dfd2f01b5",
    "d92a10c4db3974e14e88eef43f41dc4",
    "c099c763ee6e485052470de2774d6099",
    "64998426e6d48ae358dbdf2b5c6acfca",
    "5aa136c67d0a2a2852470de2774d6099",
    "9986dd19b2c459152470de2774d6099",
    "4d2d4e26349be1f3be2cbcda9b6dc9b2",
]

def create_csv():
    models = []

    for path, set in [(join(SHAPENET_DIR, 'train_test_split/shuffled_train_file_list.json'), 'train'),
                      (join(SHAPENET_DIR, 'train_test_split/shuffled_val_file_list.json'), 'val'),
                      (join(SHAPENET_DIR, 'train_test_split/shuffled_test_file_list.json'), 'test')]:
        with open(path) as f:
          files = json.load(f)
          for file in files:
              _, category_id, filename = file.split('/')
              if filename in ignore_filenames:
                  continue
              models.append({
                  'filename': filename,
                  'class': class_ids[category_id],
                  'class_id': class_ids_id[category_id],
                  'set': set,
              })

    with open(join(DEST_DIR, 'classes.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for m in models:
            writer.writerow(m.values())


def copy_files():
    with open(join(DEST_DIR, 'classes.csv'), newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            original_id = class_name_to_id[row[1]]
            filename = row[0]
            src = join(SHAPENET_MESH_DIR, original_id, filename, "models", "model_normalized.obj")
            dest = join(DEST_DIR, "data", f"{filename}.obj")
            if isfile(dest):
                continue
            if not isfile(src):
                print("not found", original_id, filename)
                continue
            copyfile(src, dest)

create_csv()
#copy_files()

# ignore theses:
# not found 02958343 6d619fcbceaa0327104b57ee967e352c
# not found 02958343 92d0fa7147696cf5ba531e418cb6cd7d
# not found 02958343 f59a474f2ec175eb7cdba8f50ac8d46c
# not found 02958343 5b04b836924fe955dab8f5f5224d1d8a
# not found 02958343 4253a9aac998848f664839bbd828e448
# not found 02958343 5973afc979049405f63ee8a34069b7c5
# not found 02958343 4ddef66f32e1902d3448fdcb67fe08ff
# not found 02958343 a8dde04ca72c5bdd6ca2b6e5474aad11
# not found 02958343 8b68f086176443b8128fe65339f3ddb2
# not found 02958343 c7bf88ef123ed4221694f51f0d69b70d
# not found 02958343 8843d862a7545d0d96db382b382d7132
# not found 02958343 a18fd5cb2a9d01c4158fe40320a23c2
# not found 02958343 29a4e6ae1f9cecab52470de2774d6099
# not found 02958343 a81cb450ce415d45bdb32c3dfd2f01b5
# not found 02958343 d92a10c4db3974e14e88eef43f41dc4
# not found 02958343 c099c763ee6e485052470de2774d6099
# not found 02958343 64998426e6d48ae358dbdf2b5c6acfca
# not found 02958343 5aa136c67d0a2a2852470de2774d6099
# not found 02958343 9986dd19b2c459152470de2774d6099
# not found 02958343 4d2d4e26349be1f3be2cbcda9b6dc9b2
