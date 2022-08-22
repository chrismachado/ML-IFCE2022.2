import torch


class Statistics:
    def __init__(self, ):
        pass

    def hit_rate(self, y_predicted, y_target):
        hit = torch.unique(y_predicted[y_predicted == y_target], return_counts=True)[1].sum()

        return hit / y_target.size(0)

