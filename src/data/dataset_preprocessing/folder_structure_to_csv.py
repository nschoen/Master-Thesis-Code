import os
import csv
from shutil import copyfile

'''
This script reads a folder structured datasets where train/test sets and classes are separated into
different folders, and moves them into a single folder and creates a csv file with information about the train/test split
and classes.
'''

def convert_set_classes_folder_structure_to_csv_folder(dirname, target_dir):
    '''
    Converts a folder structured dataset with the hierarchy test/train > classname > filename
    to a single folder with a csv file containing all set and class realted information.
    :param dirname: folder containing the folder structured dataset
    :param target_dir: target directoy to save all files in
    :return:
    '''
    class_id_counter = -1
    name_to_id = {}

    os.makedirs(target_dir, exist_ok=True)

    with open(os.path.join(target_dir, '..', 'mcb_a_classes.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')

        for set_name in ['train', 'test']:
            set_dirname = os.path.join(dirname, set_name)
            for target in sorted(os.listdir(set_dirname)):
                d = os.path.join(set_dirname, target)
                if not os.path.isdir(d):
                    continue

                if target in name_to_id:
                    class_id = name_to_id[target]
                else:
                    class_id_counter += 1
                    name_to_id[target] = class_id_counter
                    class_id = class_id_counter

                for root, _, fnames in sorted(os.walk(d)):
                    for filename in sorted(fnames):
                        ignore_files = ['00010798.obj', '00010789.obj']
                        if filename in ignore_files:
                            print("ignore", filename)
                            continue
                        src = os.path.join(d, filename)
                        dest = os.path.join(target_dir, filename)
                        copyfile(src, dest)
                        writer.writerow([filename, target, class_id, set_name])



if __name__ == '__main__':
    src = os.path.abspath('../../../datasets/mcb_a/folder_structure')
    dest = os.path.abspath('../../../datasets/mcb_a/data')
    convert_set_classes_folder_structure_to_csv_folder(src, dest)