import torch
from util.Statistics import Statistics


class PlotDecisionBoundary:
    def __init__(self, x_test, y_test, x_train, y_train, attr=None):
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train

        if attr is None or len(attr) != 2:
            raise Exception("Only two attributes are allowed.")
        self.attr = attr

        samples = torch.cat((x_train, x_test), dim=0)

        self.x1, self.x2 = torch.meshgrid([
            torch.arange(torch.min(samples[:, attr[0]]), torch.max(samples[:, attr[0]]), 0.0025),
            torch.arange(torch.min(samples[:, attr[1]]), torch.max(samples[:, attr[1]]), 0.0025)
        ], indexing='xy')

        self.x1 = self.x1.reshape(self.x1.shape[0] * self.x1.shape[1])
        self.x2 = self.x2.reshape(self.x2.shape[0] * self.x2.shape[1])

        p = []
        for e in zip(self.x1, self.x2):
            p.append(torch.tensor([e[0], e[1]]).tolist())
        self.p = torch.tensor(p, dtype=float)

    def plot_decision_boundary(self, plt,  clf, **kwargs):
        if clf.__name__ == 'KNN':
            if 'k' not in kwargs.keys():
                raise KeyError("Missing K value to run KNN.")
            _clf = self._plot_knn(clf, kwargs['k'])
        elif clf.__name__ == 'NCC':
            _clf = self._plot_ncc(clf)
        else:
            raise Exception("clf should be a valid classifier.")

        _p_targets = _clf.predict(self.p)

        ssLabel = "hit = %d%%" % (Statistics().hit_rate(_clf.predict(self.x_test[:, self.attr]), self.y_test) * 100)
        #
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.scatter(self.x1, self.x2, c=_p_targets, cmap='binary')
        plt.scatter(self.x_test[:, self.attr[0]], self.x_test[:, self.attr[1]], c=self.y_test, cmap='cool')
        plt.annotate(ssLabel, fontsize=12, xy=(0.05, 0.9), xycoords='axes fraction',
                     bbox=dict(boxstyle="round", fc=(1, .8, 0.6), ec="none"))

        return plt

    def _plot_knn(self, clf, k):
        return clf(k, self.x_train[:, self.attr], self.y_train)

    def _plot_ncc(self, clf):
        return clf(self.x_train[:, self.attr], self.y_train)
