import numpy as np


def vertebral_column(path=None):
    if not path:
        path = '../data/column_2C.dat'
    with open(path, "r") as f:
        _lines = [line.split(" ") for line in f.readlines()]
        data = []
        target = []
        for _line in _lines:
            for line in _line:
                if line == 'AB\n':
                    target.append(1)
                elif line == 'NO\n':
                    target.append(0)
                else:
                    data.append(float(line))

        data = np.reshape(data, (310, 6))
        return data, np.array(target)


def artificial_one(n, noise=0.2):
    _noises_p1 = np.random.uniform(-noise, noise, (n, 2))
    _noises_p2 = np.random.uniform(-noise, noise, (n, 2))
    _noises_p3 = np.random.uniform(-noise, noise, (n, 2))
    _noises_p4 = np.random.uniform(-noise, noise, (n, 2))

    _ones = np.ones((n, 1))
    _zeros = np.zeros((n, 1))

    _cls_100 = np.zeros((n, 2)) + _noises_p1
    _cls_101 = np.reshape(np.stack((_zeros, _ones), 2), (n, 2)) + _noises_p2
    _cls_110 = np.reshape(np.stack((_ones, _zeros), 2), (n, 2)) + _noises_p2
    _cls_111 = np.ones((n, 2)) + _noises_p4

    _cls_0s = _cls_100.shape[0] + _cls_101.shape[0] + _cls_110.shape[0]
    _cls_1s = _cls_111.shape[0]

    samples = np.concatenate((_cls_100, _cls_101, _cls_110, _cls_111))
    targets = np.concatenate((np.zeros((_cls_0s,), dtype=int), np.ones((_cls_1s,), dtype=int)))

    return samples, targets


def train_test_split(samples, targets, size):
    tt_size = int(targets.shape[0] * size)
    st_size = int(samples.shape[0] * size)

    return samples[:st_size, :], samples[st_size:, :], targets[:tt_size], targets[tt_size:]
