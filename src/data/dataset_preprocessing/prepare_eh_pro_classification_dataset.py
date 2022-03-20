import os
import csv
import numpy as np
import random

class_models = {}
class_count = {}

classes = {
    '70CC00_o_ring': 'oring',
    '70CE00_housing': 'housing',
    '70CE10_cover': 'cover',
    '70CF02_flange': 'flange',
}

category_idx_map = {
    'cover': 0,
    'flange': 1,
    'housing': 2,
    'oring': 3,
}

folder_dir = '../../datasets/eh_pro/data-class-folders'
csv_dir = '../../datasets/eh_pro'

# for _dirpath, dirnames, _filenames in os.walk(folder_dir):
#     for dir in dirnames:
#         category = classes[dir]
#         class_models[category] = []
#         for _dirpath, _dirnames, filenames in os.walk(os.path.join(folder_dir, dir)):
#             for filename in filenames:
#                 class_models[category].append({
#                     'category': category,
#                     'filename': filename
#                 })
#
# classmodels = []
# for classname, models in class_models.items():
#     n_models = len(models)
#     n_select = round(n_models * 0.8)
#     idx = list(np.random.choice(n_models, n_select, replace=False))
#     for id in range(n_models):
#         set = 'test'
#         if id in idx:
#             set = 'train'
#         classmodels.append({
#             'filename': models[id]['filename'],
#             'class': models[id]['category'],
#             'class_id': category_idx_map[models[id]['category']],
#             'set': set,
#         })
#
# with open(os.path.join(csv_dir, 'classes.csv'), 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=';')
#     for m in classmodels:
#         writer.writerow(m.values())


for _dirpath, dirnames, _filenames in os.walk(folder_dir):
    for dir in dirnames:
        category = classes[dir]
        class_models[category] = {}
        class_count[category] = 0
        for _dirpath, _dirnames, filenames in os.walk(os.path.join(folder_dir, dir)):
            for filename in filenames:
                doc_id = os.path.basename(filename).split('-')[0]
                if doc_id not in class_models[category].keys():
                    class_models[category][doc_id] = []
                class_models[category][doc_id].append({
                    'category': category,
                    'filename': filename,
                    'cls_index': category_idx_map[category],
                })
                class_count[category] += 1

models = []

for classname, documents in class_models.items():
    n_models = class_count[classname] #len(models)
    n_min_select = round(n_models * 0.78)
    n_selected = 0

    # for doc_id, docs in documents.items():y
    #     if doc_id

    while n_selected < n_min_select:
        doc_id = random.choice(list(documents.keys()))
        n_selected += len(documents[doc_id])
        for doc in documents[doc_id]:
            models.append([doc['filename'], doc['category'], doc['cls_index'], 'train'])
        documents.pop(doc_id, None)

    for doc_id, docs in documents.items():
        for d in docs:
            models.append([d['filename'], d['category'], d['cls_index'], 'test'])

# for classname, models in class_models.items():
#     n_models = len(models)
#     n_select = round(n_models * 0.8)
#     idx = list(np.random.choice(n_models, n_select, replace=False))
#     for id in range(n_models):
#         set = 'test'
#         if id in idx:
#             set = 'train'
#         classmodels.append({
#             'filename': models[id]['filename'],
#             'class': models[id]['category'],
#             'class_id': category_idx_map[models[id]['category']],
#             'set': set,
#         })


with open(os.path.join(csv_dir, 'classes.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    for m in models:
        writer.writerow(m) #.values())
