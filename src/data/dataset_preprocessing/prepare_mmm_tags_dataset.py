import csv
import numpy as np

class_models = {}

classes = ['4-drill-holes', '8-drill-holes']
# classes = ['4-drill-holes', '8-drill-holes', '12-drill-holes']

with open('../dataset_eh_mmm/flange-tags.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if len(row[2]) == 0:
            continue

        cls = -1
        tags = []
        for tag in row[2].split(','):
            if tag in classes:
                cls = tag
            tags.append(tag)

        def has_tag(stag):
            return stag in tags

        if has_tag('attached-sensor') or (has_tag('attached') and not has_tag('attached-cylinder')) or has_tag('bottom-cylinder') or has_tag('reconsider'):
            continue

        if cls == -1:
            continue

        if cls not in class_models:
            class_models[cls] = []
        class_models[cls].append({
            'category': cls,
            'category_idx': classes.index(cls),
            'filename': f"{row[0]}.stl"
        })

print("4", len(class_models['4-drill-holes']))
print("8", len(class_models['8-drill-holes']))
# print("12", len(class_models['12-drill-holes']))


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
            'class_id': models[id]['category_idx'],
            'set': set,
            'core': 'core',
        })


with open('../dataset_eh_mmm/classes-4-vs-8-drills-simple.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    for m in classmodels:
        writer.writerow(m.values())
