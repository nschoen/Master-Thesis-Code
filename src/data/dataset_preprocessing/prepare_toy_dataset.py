# only contains code for the files added on later
# see C:\Users\i00110578\projects\AIAx-Use-Case-1\datasets\MMM\flange-custom-created for the original code

import os
import csv

#train_dir = r'C:\Users\i00110578\projects\AIAx-Use-Case-1\datasets\MMM\flange-custom-created\final_flange-custom\train'
#test_dir = r'C:\Users\i00110578\projects\AIAx-Use-Case-1\datasets\MMM\flange-custom-created\final_flange-custom\test'
#csv_path = r'C:\Users\i00110578\projects\AIAx-Use-Case-1\datasets\MMM\flange-custom-created\final_flange-custom\toy.csv'

train_dir = r'C:\Users\i00110578\projects\local_datasets\toy3\train'
test_dir = r'C:\Users\i00110578\projects\local_datasets\toy3\test'
csv_path = r'C:\Users\i00110578\projects\local_datasets\toy3\toy.csv'

models = []
class_count = {}
test_class_count = {}
x1_max = 15
x1_count = 0

def read_files(dir, set):
    global x1_count
    for dirpath, _dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if not filename.endswith('.stl'):
                continue

            num_holes = filename.split('.')[0].split('-')[1]

            if num_holes == 'x1':
                cls = 'b1'
                if x1_count >= x1_max:
                    continue
                x1_count += 1
            else:
                cls = num_holes

            if cls not in class_count:
                class_count[cls] = 0
                test_class_count[cls] = 0

            class_count[cls] += 1
            if set == 'test':
                test_class_count[cls] += 1
            models.append({
                'filename': filename,
                'class': cls,
                'class_id': cls.replace('b', ''),
                'set': set,
            })

read_files(train_dir, 'train')
read_files(test_dir, 'test')

for cls, count in class_count.items():
    print('class', cls, count)
    print('train', cls, count - test_class_count[cls])
    print('test', cls, test_class_count[cls])

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    for m in models:
        writer.writerow(m.values())

# count total
# class b0 56
# train b0 53
# test b0 3
# class b1 65
# train b1 57
# test b1 8
# class b2 70
# train b2 57
# test b2 13
# class b3 74
# train b3 60
# test b3 14
# class b4 76
# train b4 61
# test b4 15
# class b5 78
# train b5 64
# test b5 14
# class b6 79
# train b6 64
# test b6 15
# class b7 79
# train b7 64
# test b7 15
# class b8 80
# train b8 64
# test b8 16
