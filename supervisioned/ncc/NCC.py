import torch


# Distância Mínima ao Centroide (DMC)
# não pode usar biblioteca para o computo da distância
# implementar disTância euclidiana
# matriz de confusão
# K em realação a Acurácia no KNN
class NCC(object):
    def __init__(self, samples, targets):
        n_samples, n_features = samples.shape
        self._cls = targets.unique()
        self.n_cls = len(self._cls)
        self._centroids = torch.zeros((self.n_cls, n_features))

        for cls in self._cls:
            self._centroids[cls] = torch.mean(samples[cls == targets], dim=0)

    def predict(self, x):
        preds = torch.zeros(x.shape[0], self.n_cls)
        for i, _x in enumerate(x):
            preds[i] = torch.tensor([torch.linalg.norm(_x - centroid) for centroid in self._centroids])
        return torch.argmin(preds, dim=1)


