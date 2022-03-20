import csv
import os
import shutil

ROOT = '../../../datasets/eh_pro'
SOURCE_ROOT = os.path.join(ROOT, 'data-obj-simplified')
DESTINATION_ROOT = os.path.join(ROOT, 'data-obj-folder-structured-simplified')

os.makedirs(DESTINATION_ROOT, exist_ok=True)

with open(os.path.join(ROOT, 'classes.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        file_base_name = row[0].split('.')[0]
        class_name = row[1]
        set_name = row[3]

        class_folder = os.path.join(DESTINATION_ROOT, class_name, set_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder, exist_ok=True)

        src = os.path.join(SOURCE_ROOT, f"{file_base_name}.obj")
        dest = os.path.join(class_folder, f"{file_base_name}.obj")
        shutil.copyfile(src, dest)