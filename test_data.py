import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns


class test_data:
    def __init__(self):
        pass

    @staticmethod
    def get_mnist():
        """ Работает при наличии суперкомпьютера """
        MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
        X_train = MNIST_train.data
        targets = MNIST_train.targets
        X_train = X_train.float()
        X_train = X_train.reshape([-1, 28*28])

        data = X_train.numpy()[:5000, :]
        targets = targets.numpy()[:5000]
        return data, targets

    @staticmethod
    def get_spheres():
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x1 = (np.cos(u) * np.sin(v)).reshape((-1, 1))
        y1 = (np.sin(u) * np.sin(v)).reshape((-1, 1))
        z1 = (np.cos(v)).reshape((-1, 1))

        x2 = x1 + 2
        y2 = y1 + 2
        z2 = z1 + 2

        x = np.concatenate([x1, x2], axis=0)
        y = np.concatenate([y1, y2], axis=0)
        z = np.concatenate([z1, z2], axis=0)
        data = np.concatenate([x, y, z], axis=1)
        return data

    @staticmethod
    def get_add_sphere():
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x1 = (np.cos(u) * np.sin(v)).reshape((-1, 1)) - 2
        y1 = (np.sin(u) * np.sin(v)).reshape((-1, 1)) - 2
        z1 = (np.cos(v)).reshape((-1, 1)) - 2

        data = np.concatenate([x1, y1, z1], axis=1)
        return data

    @staticmethod
    def get_add_sphere_2():
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x1 = (np.cos(u) * np.sin(v)).reshape((-1, 1)) + 2
        y1 = (np.sin(u) * np.sin(v)).reshape((-1, 1)) - 2
        z1 = (np.cos(v)).reshape((-1, 1)) + 2

        data = np.concatenate([x1, y1, z1], axis=1)
        return data

    @staticmethod
    def get_add_sphere_3():
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x1 = (np.cos(u) * np.sin(v)).reshape((-1, 1)) / 2
        y1 = (np.sin(u) * np.sin(v)).reshape((-1, 1)) / 2
        z1 = (np.cos(v)).reshape((-1, 1)) / 2

        data = np.concatenate([x1, y1, z1], axis=1)
        return data

    @staticmethod
    def visualize_3d(data):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        plt.show()

    @staticmethod
    def visualize_heatmap(data):
        fig = plt.figure()
        ax = fig.add_subplot()

        sns.heatmap(data, ax=ax, cmap="rainbow")

        fig.set_figwidth(14)
        fig.set_figheight(6)

        plt.show()

    @staticmethod
    def normalize_minmax(data):
        """ Использовать для изображений """
        min_val = 0  # Нижняя граница
        max_val = 1  # Верхняя граница
        norm_data = preprocessing.minmax_scale(data, (min_val, max_val))
        return norm_data