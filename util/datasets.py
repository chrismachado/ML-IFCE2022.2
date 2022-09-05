import torch
import matplotlib.pyplot as plt

def vertebral_column(path=None):
    if not path:
        path = 'data/column_2C.dat'
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

        data = torch.reshape(torch.tensor(data), (310, 6))
        return data, torch.tensor(target, dtype=int)


def artificial_i(n, noise=0.2):
    _noises_p1 = torch.distributions.uniform.Uniform(-noise, noise).sample([n, 2])
    _noises_p2 = torch.distributions.uniform.Uniform(-noise, noise).sample([n, 2])
    _noises_p3 = torch.distributions.uniform.Uniform(-noise, noise).sample([n, 2])
    _noises_p4 = torch.distributions.uniform.Uniform(-noise, noise).sample([n, 2])

    _ones = torch.ones((n, 1))
    _zeros = torch.zeros((n, 1))

    _cls_100 = torch.zeros((n, 2))
    _cls_101 = torch.stack((_zeros, _ones), 2)
    _cls_111 = torch.ones((n, 2))
    print(_cls_101.shape)

    # samples = torch.rand((n, 2))
    # return torch.cat((_cls_100))
    return torch.cat((_cls_100, _cls_111))


if __name__ == '__main__':
    s = artificial_i(25)
    # print(s)

    plt.scatter(s[:, 0], s[:, 1])
    plt.show()




