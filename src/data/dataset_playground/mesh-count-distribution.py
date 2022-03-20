import open3d as o3d
from os import walk
from os.path import join
import csv
import math

def count_edges(mesh):
    mesh.compute_adjacency_list()
    edge_count = 0
    # i = 0
    for i, al in enumerate(mesh.adjacency_list):
        for r in al:
            if i < r:
                edge_count += 1
    return edge_count

min_edge = 100000000000000
max_edge = 0

distribution = {}

dir = r'C:\Users\i00110578\projects\local_datasets\shapenet\data-2000-no-triangulate'

check_model_set = True
if check_model_set:
    model_set = {}
    distribution_test = {}
    distribution_train = {}
    with open(r'C:\Users\i00110578\projects\local_datasets\shapenet\classes.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            basename = row[0].split('.')[0]
            model_set[basename] = row[3]

#for (dirpath, dirnames, filenames) in walk("../../../datasets/eh_pro/data-obj"):
for (dirpath, dirnames, filenames) in walk(dir):
    for filename in filenames:
        if ".obj" not in filename:
            continue
        # print("filename", filename)
        basename = filename.split('.')[0]
        mesh = o3d.io.read_triangle_mesh(join(dirpath, filename))
        edge_count = count_edges(mesh)
        if edge_count > max_edge:
            max_edge = edge_count
        if edge_count < 100:
            print("below 100 edges", filename, edge_count)

        if edge_count > 10000:
            edge_count = math.ceil(float(edge_count) / 1000.0) * 1000
        else:
            edge_count = math.ceil(float(edge_count) / 100.0) * 100

        if edge_count not in distribution:
            distribution[edge_count] = 0
        distribution[edge_count] = distribution[edge_count] + 1
        if check_model_set and basename in model_set:
            if model_set[basename] == 'train':
                if edge_count not in distribution_train:
                    distribution_train[edge_count] = 0
                distribution_train[edge_count] = distribution_train[edge_count] + 1
            else:
                if edge_count not in distribution_test:
                    distribution_test[edge_count] = 0
                distribution_test[edge_count] = distribution_test[edge_count] + 1

# {'5000': 143, '10000': 249, '15000': 47, '20000': 30, '25000': 40, '30000': 55, '35000': 79, '40000': 82, '45000': 24, '50000': 98, 'above': 414}
# max_edge 216366

# {'5000': 143, '10000': 249, '15000': 47, '20000': 30, '25000': 40, '30000': 55, '35000': 79, '40000': 82, '45000': 24,
# '50000': 98, '80000': 241, '100000': 73, '120000': 27, '140000': 13, '160000': 23, '180000': 14, '200000': 12, '210000': 8, 'above': 3}
# max_edge 216366

print(distribution)
print("max_edge", max_edge)

if check_model_set:
    print('test', distribution_test)
    print('train', distribution_train)


# local_datasets/mcb_a/data-2048
# below 100 edges 00010798.obj 0 => Removed from csv list
# below 100 edges 00010789.obj 0 => Removed from csv list
# below 100 edges 00027538.obj 44
# below 100 edges 00048649.obj 28
# {'10': 1, '100': 2, '500': 1669, '5000': 56960, '10000': 24, '15000': 8, '20000': 10,
# '25000': 4, '30000': 4, '35000': 3, '40000': 2, '45000': 1, '50000': 4, '80000': 1,
# '100000': 0, '120000': 0, '140000': 1, '160000': 1, '180000': 0, '200000': 0, '210000': 0, 'above': 0}
# max_edge 145488