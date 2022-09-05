import numpy as np


def hitrate(target, predicted):
    _hitrate = np.unique(predicted[predicted == target], return_counts=True)[1].sum()

    return _hitrate / target.shape[0]


def confusion_matrix(targets, predicted, n_cls):
    cfs_matrix = np.zeros((n_cls, n_cls))

    for i in range(targets.shape[0]):
        cfs_matrix[predicted[i], targets[i]] += 1

    return cfs_matrix


def accuracy(arr):
    if arr.shape[1] != 1:
        _means = []
        for i in range(arr.shape[1]):
            _means.append(np.mean(arr[:, i]))
        return np.asarray(_means)
    return np.mean(arr)


def standard_deviation(arr):
    if arr.shape[1] != 1:
        std = []
        for i in range(arr.shape[1]):
            std.append(np.std(arr[:, i]))
        return np.asarray(std)
    return np.std(arr)


def argsminmax(arr, arg_func):
    if arg_func is None:
        raise Exception('Should pass a max or min function.')

    if arr.shape[1] != 1:
        args = []
        for i in range(arr.shape[1]):
            args.append(arg_func(arr[:, i]))
        return np.asarray(args)
    return arg_func(arr)


def normalize(x):
    n_features = x.shape[1]
    _normalized = []
    for i in range(n_features):

        _max = x[:, i].max()
        _min = x[:, i].min()

        _normalized.append((x[:, i] - _min) / (_max - _min))

    return np.array(_normalized).transpose()
