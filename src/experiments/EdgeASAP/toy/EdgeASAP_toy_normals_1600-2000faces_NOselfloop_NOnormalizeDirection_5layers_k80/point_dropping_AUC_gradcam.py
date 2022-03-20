"""
PointNet++ Grad-CAM Area under the curve (AUC) retrieval.
Loads the mat-*.npz file which is a product of a point dropping experiment.
It takes all measured points and calculates the AUC as a fraction of the highest possible value, i.e.
a curve with 100% accuracy with all measured points.
"""

from src.utils.point_dropping_experiment import PointDropExperiment
from os.path import dirname, join


if __name__ == '__main__':
    num_drops = 3000
    exp = PointDropExperiment(module=None, test_dataset=None, num_drops=num_drops, steps=100, classifier=None,
                              grad_cam=None, testdataloader=None)

    mat_path = join(dirname(__file__), 'results/point_dropping/gradcam-2300drops-noHU-113pcds-3000drops-100steps.npz')
    exp.load_by_npz_name(mat_path)
    exp.num_drops = num_drops
    auc = exp.get_area_under_the_curve()

    print('EdgeASAP Mesh Grad-CAM no-heatmap-updates 3000 drops')
    print("AUC", auc)
    print('random total:', auc[0][0])
    print('low-drop total:', auc[0][1])
    print('high-drop total:', auc[0][2])
    print('random total between 0.0 and 1.0:', auc[0][0] / num_drops)
    print('low-drop total between 0.0 and 1.0:', auc[0][1] / num_drops)
    print('high-drop total between 0.0 and 1.0:', auc[0][2] / num_drops)
    # print('low-drop total between 0.92 and 0.99:', (auc[0][1] - (num_drops * 0.92)) / ((0.99 - 0.92) * num_drops))
    # print('high-drop total between 0.825 and 0.99:', (auc[0][2] - (num_drops * 0.825)) / ((0.99 - 0.825) * num_drops))

    # Result:
    # AUC [[ 1981.4159292   1246.01769912   960.17699115]
    #      [ 1755.92796613  1141.36539645   884.43251584]
    #      [ 3329.39176981 12667.27958612 14675.09616413]]
    # random total: 1981.41592920354
    # low-drop total: 1246.0176991150443
    # high-drop total: 960.1769911504427
    # random total between 0.0 and 1.0: 0.6604719764011799
    # low-drop total between 0.0 and 1.0: 0.4153392330383481
    # high-drop total between 0.0 and 1.0: 0.32005899705014756