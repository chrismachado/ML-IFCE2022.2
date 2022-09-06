import numpy as np
import matplotlib.pyplot as plt
from util.metrics import confusion_matrix
import seaborn as sns


def plot_decision_boundary(clf, x, target, pos=111):
    if not x.shape[1] == 2:
        raise Exception("Samples dimensions should be 2.")

    plt.subplot(pos)
    surface = __x_surface_01()
    classified_surface = clf.predict(surface)

    plt.scatter(surface[:, 0], surface[:, 1], c=classified_surface, cmap=plt.cm.RdYlBu, alpha=.25)
    plt.scatter(x[:, 0], x[:, 1], c=target, cmap=plt.cm.RdYlBu)

    plt.title(clf)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.xlim(0, 1)
    plt.ylim(0, 1)


def __x_surface_01():
    x1, x2 = np.meshgrid(np.arange(0, 1, 0.0125),
                         np.arange(0, 1, 0.0125),
                         indexing='xy')
    x1 = x1.reshape((x1.shape[0] * x1.shape[1],))
    x2 = x2.reshape((x2.shape[0] * x2.shape[1],))

    _surface = []

    for p in zip(x1, x2):
        _surface.append([p[0], p[1]])

    return np.array(_surface)


def plot_confusion_matrix(clf, x, targets, pos=111):
    predicted = clf.predict(x)
    cfs_matrix = confusion_matrix(targets=targets, predicted=predicted, n_cls=clf.n_cls)
    plt.subplot(pos)

    ax = sns.heatmap(cfs_matrix, annot=True, cmap='Blues')
    ax.set_title(clf)
    ax.set_xlabel('Predicted values')
    ax.set_ylabel('Actual values')


def plot_bar(clf, x, y, pos):
    plt.subplot(pos)
    default_x_ticks = range(len(x))
    plt.bar(default_x_ticks, y, width=.5)
    plt.xticks(default_x_ticks, x)
    plt.title(clf)
    plt.xlabel('x0')
    plt.ylabel('x1')


def plot_bar_std(std, values):
    default_x_ticks = range(len(values))
    plt.bar(default_x_ticks, std, color='maroon', width=.4)
    plt.xticks(default_x_ticks, values)
    plt.xlabel('Standard Deviation')
    plt.ylabel('Values')
    plt.title('Standard Deviation')
    plt.show()


def plot_bar_acc(acc, values):
    default_x_ticks = range(len(values))
    plt.bar(default_x_ticks, acc, width=.4)
    plt.xticks(default_x_ticks, values)
    plt.ylabel('Accuracy')
    plt.xlabel('Values')
    plt.title('Accuracy')
    plt.show()
