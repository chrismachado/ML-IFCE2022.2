import torch
import numpy as np

class Statistics:
    def __init__(self, ):
        pass

    def hit_rate(self, y_predicted, y_target):
        hit = np.unique(y_predicted[y_predicted == y_target], return_counts=True)[1].sum()

        return hit / y_target.shape[0]

