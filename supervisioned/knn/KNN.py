import torch


class KNN(object):
    def __init__(self, K, samples, targets):
        self.K = K
        self.samples = samples
        self.targets = targets

    def predict(self, x):
        cdists = torch.cdist(x, self.samples)
        idxs = cdists.argsort()[:, :self.K]
        nn = self.targets[idxs]

        return nn.mode(dim=1).values

