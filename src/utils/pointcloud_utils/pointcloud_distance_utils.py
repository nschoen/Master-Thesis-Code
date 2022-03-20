import open3d as o3d
import numpy as np
import os
import time

trees = {}

# setup distance function
def chamfer_distance(pc1, pc2, index1, index2, mean=True):
    if not index1 in trees:
        trees[index1] = o3d.geometry.KDTreeFlann(pc1)
    if not index2 in trees:
        trees[index2] = o3d.geometry.KDTreeFlann(pc2)
    pc_tree1 = trees[index1]
    pc_tree2 = trees[index2]
        
    # pc_tree1 = o3d.geometry.KDTreeFlann(pc1)
    # pc_tree2 = o3d.geometry.KDTreeFlann(pc2)
    
    distance_p1_p2 = 0.0
    for p in pc1.points:
        [k, idx, _] = pc_tree2.search_knn_vector_3d(p, 1)
        nearest = pc2.points[idx[0]]
        distance_p1_p2 += np.linalg.norm(p - nearest)

    distance_p2_p1 = 0.0
    for p in pc2.points:
        [k, idx, _] = pc_tree1.search_knn_vector_3d(p, 1)
        nearest = pc1.points[idx[0]]
        distance_p2_p1 += np.linalg.norm(p - nearest)
    
    if mean:
        return distance_p1_p2/len(pc1.points) + distance_p2_p1/len(pc2.points)
    else:
        return distance_p1_p2 + distance_p2_p1

def calculate_similarity_matrix(models):
    n_models = len(models)
    simi_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        j = i + 1
        start = time.time()
        
        while j < n_models:
            if simi_matrix[i, j] > 0 and simi_matrix[j, i] > 0:
                j += 1
                continue
                
            dis = chamfer_distance(models[i]['pcd'], models[j]['pcd'], i, j)
            simi_matrix[i, j] = dis
            simi_matrix[j, i] = dis
            j += 1
            
        print("Done in", time.time() - start, i)
    
    return simi_matrix

def save_similarity_matrix(file, matrix):
    if not file.endswith('.npy'):
        file = f"{file}.npy"
    np.save(file, matrix)

def load_similarity_matrix(file):
    if not file.endswith('.npy'):
        file = f"{file}.npy"
    return np.load(file)