from __future__ import print_function
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import numpy as np
#import neptune
#import wandb

def save_confusion_matrix(y_true, y_pred, classes, fig_destination, save_conf_in_wandb=False):
    cm = confusion_matrix(y_true, y_pred)
    class_names = classes

    # print("cm", cm)
    # print("sm", np.sum(cm, axis=1).reshape((7,1)))
    # percentage = cm / np.sum(cm, axis=1).reshape((7,1))
    # print("percentage", percentage)
    # print("donw")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(include_values=True, cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal')
    plt.savefig(os.path.join(fig_destination, 'confusion-matrix.png'))
    #if save_conf_in_wandb:
    #    wandb.log({set + "-confusion-matrix": os.path.join(fig_destination, 'confusion-matrix.png')})
    #neptune.log_image('confusion_matrix', os.path.join(fig_destination, 'confusion-matrix.png'))

    cm = cm / cm.astype(np.float).sum(axis=1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names,)
    disp.plot(include_values=True, cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal')
    #if save_conf_in_wandb:
    #    plt.savefig(os.path.join(fig_destination, 'confusion-matrix-percentage.png'))
    #wandb.log({set + "-confusion-matrix-percentage": os.path.join(fig_destination, 'confusion-matrix-percentage.png')})
    #neptune.log_image('confusion_matrix_percent', os.path.join(fig_destination, 'confusion-matrix-percentage.png'))
    plt.clf()
    plt.close()