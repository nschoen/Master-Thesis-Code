import csv

#CSV_FILE = r'C:\Users\i00110578\projects\local_datasets\toy-final\toy.csv'
#CSV_FILE = r'C:\Users\i00110578\projects\local_datasets\eh_pro\classes.csv'
CSV_FILE = r'C:\Users\i00110578\projects\local_datasets\eh_mmm\classes-core.csv'

classes_count = {
    'train': {},
    'test': {},
}
total_count = {
    'train': 0,
    'test': 0,
}

with open(CSV_FILE, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        set = 'train' if row[3] == 'val' else row[3]
        classname = row[1]
        if classname not in classes_count[set]:
            classes_count[set][classname] = 0
        classes_count[set][classname] += 1
        total_count[set] += 1

print(classes_count)
print(total_count)

# toy
# {'train': {'b0': 53, 'b1': 57, 'b2': 57, 'b3': 60, 'b4': 61, 'b5': 64, 'b6': 64, 'b7': 64, 'b8': 64},
#  'test': {'b5': 14, 'b6': 15, 'b7': 15, 'b8': 16, 'b0': 3, 'b1': 8, 'b2': 13, 'b3': 14, 'b4': 15}}
# total: {'train': 544, 'test': 113}

# EH PRO
# {'train': {'flange': 202, 'cover': 286, 'oring': 146, 'housing': 355, 'adapter': 320},
#  'test': {'flange': 57, 'cover': 77, 'oring': 41, 'housing': 99, 'adapter': 80}}
# total {'train': 1309, 'test': 354}

# MMM
# {'train': {'flange': 1080, 'cover': 115, 'tube': 110, 'sensor': 244, 'housing': 170, 'mounting': 58, 'rodprobe': 96},
#  'test': {'flange': 270, 'cover': 29, 'tube': 27, 'sensor': 61, 'housing': 43, 'mounting': 15, 'rodprobe': 24}}
# total {'train': 1873, 'test': 469}