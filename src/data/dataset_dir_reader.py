import os
import csv


# def read_directory(root, config, split, cross_validation_set, mesh_dir):
#     if config.models_csv:
#         # csv provided with all filename, class, and test/train/cross_val split information
#         return read_csv_dataset_directory(root, config, split, cross_validation_set, mesh_dir)
#     else:
#         # no csv provided, use the folder structure for classes and test/train split
#         return read_folder_structure_dataset_directory(root, config, split)


def read_csv_dataset_directory(root, config, split, cross_validation_set, mesh_dirs):
    models = []
    classid_map = {}

    with open(os.path.join(root, config.models_csv), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if cross_validation_set and (
                    (split == 'train' and int(row[4]) == cross_validation_set) or (
                    split == 'test' and int(row[4]) != cross_validation_set)
            ) or not cross_validation_set and row[3] != split:
                continue

            classid_map[int(row[2])] = row[1]
            if config.filter_classes and int(row[2]) not in config.filter_classes:
                continue

            filename = row[0]
            basename = filename.split('.')[0]
            filename = f"{basename}.obj"

            if config.filename_filter and basename not in config.filename_filter:
                continue

            for mesh_dir in mesh_dirs:
                datapath = os.path.join(mesh_dir, filename)

                models.append({
                    'cls': row[1],
                    'cls_idx': int(row[2]),
                    'datapath': datapath,
                    'filename': filename,
                })

        # create sorted list of classes
        class_ids = list(classid_map.keys())
        class_ids.sort()
        classes = [classid_map[idx] for idx in class_ids]

    return models, classid_map, classes


# def read_folder_structure_dataset_directory(root, config, split):
#     assert split in ['test', 'train']
#
#     models = []
#     classid_map = {}
#
#     data_dir = os.path.join(root, split)
#     class_id = -1
#
#     for target in sorted(os.listdir(root)):
#         d = os.path.join(data_dir, target)
#         if not os.path.isdir(d):
#             continue
#
#         class_id += 1
#         classid_map[class_id] = target
#
#         if config.filter_classes and class_id not in config.filter_classes:
#             continue
#
#         for root, _, fnames in sorted(os.walk(d)):
#             for fname in sorted(fnames):
#                 basename = fname.split('.')[0]
#
#                 if config.filename_filter and basename not in config.filename_filter:
#                     continue
#
#                 models.append({
#                     'cls': target,
#                     'cls_idx': class_id,
#                     'datapath': os.path.join(d, fname),
#                     'filename': fname,
#                 })
#
#     # create sorted list of classes
#     class_ids = list(classid_map.keys())
#     class_ids.sort()
#     classes = [classid_map[idx] for idx in class_ids]
#
#     return models, classid_map, classes