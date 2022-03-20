"""
PointNet++ RISE Area under the curve (AUC) retrieval.
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

    mat_path = join(dirname(__file__), 'results/point_dropping/mat-rise-noHU-113pcds-3000drops-100steps.npz')
    exp.load_by_npz_name(mat_path)
    exp.num_drops = num_drops
    auc = exp.get_area_under_the_curve()

    print('PointNet++ Point RISE no-heatmap-updates 3000 drops')
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
    # AUC [[ 1969.02654867  1946.01769912   306.19469027]
    #     [ 1757.33683848  1783.48918932   345.53404909]
    #     [ 3319.19420293  5540.49566151 17974.98038976]]
    # random total: 1969.0265486725668
    # low-drop total: 1946.017699115044
    # high-drop total: 306.1946902654867
    # random total between 0.0 and 1.0: 0.6563421828908557      Used in Thesis
    # low-drop total between 0.0 and 1.0: 0.6486725663716814    Used in Thesis
    # high-drop total between 0.0 and 1.0: 0.10206489675516224  Used in Thesis