import numpy as np


def euclidian_distance(x1, x2):
    _cdists = 0
    cdist = []
    for _x1 in x1:
        for _x2 in x2:
            _cdists = np.sqrt(np.power(_x1 - _x2, 2).sum())
            cdist.append(_cdists)

    return np.array(cdist).reshape((x1.shape[0], x2.shape[0]))


def normalize(x):
    _max = x.max()
    _min = x.min()

    z = (x - _min) / (_max - _min)

    return z


if __name__ == '__main__':
    import numpy as np
    x1 = np.random.rand(4, 2)
    x2 = np.random.rand(3, 2)

    print(euclidian_distance(x2, x1))
