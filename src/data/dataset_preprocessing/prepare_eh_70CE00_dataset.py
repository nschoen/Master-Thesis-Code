import os
import csv
import numpy as np
import random

# with open('../datasets/eh_prop/raw-categories-70CE00-xproven.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=';')
#
#     counter = 0
#     nochamber = 0
#     chamber = 0
#     nonx = 0
#
#     for row in reader:
#         if row[1] == 'Dismiss':
#             continue
#         if row[1] == 'X' or row[1] == 'X with chamber':
#              counter += 1
#         if row[1] == 'Non-X with Chamber' or row[1] == 'X with chamber':
#              chamber += 1
#         if row[1] == 'Non-X' or row[1] == 'X':
#              nochamber += 1
#         if row[1] == 'Non-X' or row[1] == 'Non-X with Chamber':
#              nonx  += 1
#
# print(counter)
# print(nochamber)
# print(chamber)
# print(nonx)
# import sys
# sys.exit(1)

class_models = {}
class_count = {}

with open('../datasets/eh_pro/raw-categories-70CE00-xproven.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')

    for row in reader:
        cls = 'non-chamber'
        cls_index = 0
        if row[1] == 'Dismiss':
            continue
        if row[1] == 'X with chamber' or row[1] == 'Non-X with Chamber':
            cls = 'chamber'
            cls_index = 1

        if cls not in class_models.keys():
            class_models[cls] = {}
            class_count[cls] = 0
        doc_id = row[0].split('-')[0]
        if doc_id not in class_models[cls].keys():
            class_models[cls][doc_id] = []

        class_models[cls][doc_id].append({
            "name": row[0],
            "cls": cls,
            "cls_index": cls_index,
        })
        class_count[cls] += 1

models = []
# set_assignment = {}

for classname, documents in class_models.items():
    n_models = class_count[classname] #len(models)
    n_min_select = round(n_models * 0.78)
    n_selected = 0

    # for doc_id, docs in documents.items():
    #     if doc_id

    while n_selected < n_min_select:
        doc_id = random.choice(list(documents.keys()))
        n_selected += len(documents[doc_id])
        for doc in documents[doc_id]:
            models.append([doc['name'], doc['cls'], doc['cls_index'], 'train'])
        documents.pop(doc_id , None)

    for doc_id, docs in documents.items():
        for d in docs:
            models.append([d['name'], d['cls'], d['cls_index'], 'test'])

with open('../datasets/eh_pro/classes-chamber.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    for m in models:
        writer.writerow(m)




# for dirpath, _dirnames, filenames in os.walk('../dataset_eh_mmm'):
#     category = dirpath.split('/')[-1]
#     if category == 'dismiss':
#         continue
#     class_models[category] = []
#     for filename in filenames:
#         class_models[category].append({
#             'category': category,
#             'datapath': os.path.join(dirpath, filename),
#             'filename': filename
#         })
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
#             'core': 'core' if category_idx_map[models[id]['category']] <= 6 else 'extra',
#         })
#
#
# with open('../dataset_eh_mmm/classes.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=';')
#     for m in classmodels:
#         writer.writerow(m.values())
