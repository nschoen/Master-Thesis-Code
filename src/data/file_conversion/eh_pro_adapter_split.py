import os
import csv
import numpy as np
import random

filenames = []
class_id = 4
class_name = 'adapter'

SRC = r"C:\Users\i00110578\projects\local_datasets\eh_pro_adapters\stl-latest-version"
CSV_DEST = r"C:\Users\i00110578\projects\local_datasets\eh_pro_adapters"

for dirpath, _dirnames, filenames in os.walk(SRC):
    filenames = filenames

n_models = len(filenames)
n_select = round(n_models * 0.8)
idx = list(np.random.choice(n_models, n_select, replace=False))
models = []
for id in range(n_models):
    set = 'test'
    if id in idx:
        set = 'train'
    models.append({
        'filename': filenames[id],
        'class': 'adapter',
        'class_id': 4,
        'set': set,
    })

with open(os.path.join(CSV_DEST, 'adapter-latest-version.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    for m in models:
        writer.writerow(m.values())
