import os
import csv
import numpy as np

category_idx_map = {
    'cover': 0,
    'flange': 1,
    'housing': 2,
    'mounting': 3,
    'rodprobe': 4,
    'tube': 5,
    'sensor': 6,
    #'cableconnector': 7,
    #'fork': 8,
    #'funnelantenna': 9,
    #'diverse': 10,
}

class_models = {}

# for dirpath, _dirnames, filenames in os.walk('../datasets_eh_mmm'):
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

for dirpath, _dirnames, filenames in os.walk('../../../datasets/eh_mmm/data'):
    for filename in filenames:
        category = filename.split('_')[0]
        if category not in category_idx_map.keys():
            continue
        if category not in class_models.keys():
            class_models[category] = []

        class_models[category].append({
            'category': category,
            'datapath': os.path.join(dirpath, filename),
            'filename': filename
        })

classmodels = []
for classname, models in class_models.items():
    n_models = len(models)
    n_select = round(n_models * 0.8)
    idx = list(np.random.choice(n_models, n_select, replace=False))
    for id in range(n_models):
        set = 'test'
        if id in idx:
            set = 'train'
        classmodels.append({
            'filename': models[id]['filename'],
            'class': models[id]['category'],
            'class_id': category_idx_map[models[id]['category']],
            'set': set,
            'core': 'core' if category_idx_map[models[id]['category']] <= 6 else 'extra',
        })


with open('../dataset_eh_mmm/classes.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    for m in classmodels:
        writer.writerow(m.values())
