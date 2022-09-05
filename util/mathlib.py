import numpy as np


def euclidian_distance(x1, x2):
    _cdists = 0
    cdist = []
    for _x1 in x1:
        for _x2 in x2:
            _cdists = np.sqrt(np.power(_x1 - _x2, 2).sum())
            cdist.append(_cdists)

    return np.array(cdist).reshape((x1.shape[0], x2.shape[0]))

