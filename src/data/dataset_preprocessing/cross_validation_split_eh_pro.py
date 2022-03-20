import csv
import numpy as np
import random
import os

"""
Takes a classes csv file and splits the entries belonging to the train test in 5 folds randomly within each class.
For the first fold 0, the set is changed to 'val'. The valiadtion set id is appended.

This version takes the different documents versions of the eh pro dataset into account by making sure that
similar documents are in the same set split.
"""

CSV_SRC = r'C:\Users\i00110578\projects\local_datasets\eh_pro\classes-without-val-split.csv'
CSV_DEST = r'C:\Users\i00110578\projects\local_datasets\eh_pro\classes.csv'

new_entries = []
models_by_classes = {}
class_count = {}

with open(CSV_SRC, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if row[3] == 'test':
            row.append('')  # test samples will be in cross validation set i=0
            new_entries.append(row)
        else:
            doc_id = row[0].split('-')[0]
            class_id = row[2]
            if not class_id in class_count:
                class_count[class_id] = 0
            if not class_id in models_by_classes:
                models_by_classes[class_id] = {}
            if not doc_id in models_by_classes[class_id]:
                models_by_classes[class_id][doc_id] = []
            models_by_classes[class_id][doc_id].append(row)
            class_count[class_id] += 1

for class_id, documents in models_by_classes.items():
    n_models = class_count[class_id]
    n_target_select = round(n_models * 0.2)
    n_rest_models = n_models

    for j in range(5):
        n_selected = 0
        n_min_select = n_target_select if j < 4 else n_rest_models

        while n_selected < n_min_select:
            doc_id = random.choice(list(documents.keys()))
            n_selected += len(documents[doc_id])
            for doc in documents[doc_id]:
                if j == 0:
                    doc[3] = 'val'
                doc.append(f"{j}")
                new_entries.append(doc)
            documents.pop(doc_id, None)

        n_rest_models -= n_selected

with open(CSV_DEST, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    for m in new_entries:
        writer.writerow(m)
