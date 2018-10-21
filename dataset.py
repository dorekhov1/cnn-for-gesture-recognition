'''
    Extend the torch.utils.data.Dataset class to build a GestureDataset class.
'''

import torch.utils.data as data
import numpy as np


def generate_noisy_data(X, eps=1e-2):
    return np.random.normal(scale=eps, size=X.shape) + X


class GestureDataset(data.Dataset):

    def __init__(self, X, y, add_noise=True, noise_eps=1e-2, noisy_samples=100000):
        self.X = [X]
        self.y = [y]

        if add_noise:
            while noisy_samples > 0:
                noisy_features = generate_noisy_data(X, eps=noise_eps)

                self.X.append(noisy_features)
                self.y.append(y)

                noisy_samples -= len(noisy_features)

        self.X = np.array(self.X).reshape((-1, *X.shape[1:]))
        self.y = np.array(self.y).reshape((-1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
