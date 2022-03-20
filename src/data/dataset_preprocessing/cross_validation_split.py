import csv
import numpy as np
import random

"""
Takes a classes csv file and splits the entries belonging to the train test in 5 folds randomly within each class.
For the first fold 0, the set is changed to 'val'. The valiadtion set id is appended.
"""

#CSV_SRC = '../../../datasets/eh_mmm_custom_flanges/toy9.csv'
#CSV_DEST = '../../../datasets/eh_mmm_custom_flanges/toy9_cross_val.csv'

#CSV_SRC = '../../../datasets/mcb_a/mcb_a_classes.csv'
#CSV_DEST = '../../../datasets/mcb_a/claseses.csv'

#CSV_SRC = '../../../datasets/eh_mmm/classes-core.csv'
#CSV_DEST = '../../../datasets/eh_mmm/classes-core-val.csv'

#CSV_SRC = '../../../datasets/eh_mmm_custom_flanges/toy9.csv'
#CSV_DEST = '../../../datasets/eh_mmm_custom_flanges/toy9-val.csv'

#CSV_SRC = '../../../datasets/eh_pro/classes.csv'
#CSV_DEST = '../../../datasets/eh_pro/classes-val.csv'

CSV_SRC = r'C:\Users\i00110578\projects\local_datasets\toy3\toy.csv'
CSV_DEST = r'C:\Users\i00110578\projects\local_datasets\toy3\toy-validation-split.csv'

new_entries = []
models_by_classes = {}

with open(CSV_SRC, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if row[3] == 'test':
            row.append('')  # test samples will be in cross validation set i=0
            new_entries.append(row)
        else:
            if not row[2] in models_by_classes:
                models_by_classes[row[2]] = []
            models_by_classes[row[2]].append(row)

for class_id, models in models_by_classes.items():
    n_models = len(models)
    n_select = round(n_models * 0.15)

    for j in range(5):
        num_select = n_select if j < 4 else n_models
        idx = list(np.random.choice(n_models, num_select, replace=False))
        #selected_models = random.sample(models, n_select)
        #models = [i for j, i in enumerate(somelist) if j not in indices]

        for id in idx:
            model = models[id]
            if j == 0:
                model[3] = 'val'
            model.append(f"{j}")  # test samples will be in cross validation set i=0
            new_entries.append(model)

        models = [i for j, i in enumerate(models) if j not in idx]
        n_models -= n_select

with open(CSV_DEST, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    for m in new_entries:
        writer.writerow(m)
