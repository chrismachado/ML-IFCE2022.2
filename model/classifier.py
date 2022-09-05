import numpy as np
from util.mathlib import euclidian_distance
from scipy import stats


class KNN:
    def __init__(self, k, samples, targets):
        self._k = 1  # minimum value
        self._samples = []
        self._targets = []
        self.n_cls = 0
        self.fit(k, samples, targets)

    def fit(self, k, samples, targets):
        self._k = k
        self._samples = samples
        self._targets = targets
        self.n_cls = len(np.unique(targets))

    def predict(self, x):
        distance = euclidian_distance(x, self._samples)
        idxs = distance.argsort()[:, :self._k]
        knn = self._targets[idxs]
        return stats.mode(knn, axis=1, keepdims=False)[0]

    def __str__(self):
        return 'K-Nearest Neighbors [k=%d, n_cls=%d]' % (self._k, len(np.unique(self._targets)))


class NCC:
    def __init__(self, samples, targets):
        self.centroids = []
        self.n_cls = 0
        self.fit(samples, targets)

    def fit(self, samples, targets):
        n_samples, n_features = samples.shape
        _cls = np.unique(targets)
        self.n_cls = len(_cls)
        _cents = np.zeros((self.n_cls, n_features))

        for cls in _cls:
            _cents[cls] = np.mean(samples[cls == targets], axis=0)

        self.centroids = _cents
        return self.centroids

    def predict(self, x):
        distance = euclidian_distance(x, self.centroids)
        return distance.argmin(axis=1)

    def __str__(self):
        return 'Nearest Centroid Classifier [n_cls=%d]' % len(self.centroids)
