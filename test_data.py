import torchvision.datasets
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, datasets, decomposition
from sklearn.datasets import make_blobs
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
    def get_sphere(x_offset=0, y_offset=0, z_offset=0, volume=1):
        """
        Возвращает 200 точек сферы. Каждая строка - координаты x, y, z точки.
        :param x_offset: смещение координат по оси x (от нуля)
        :param y_offset: смещение координат по оси y (от нуля)
        :param z_offset: смещение координат по оси z (от нуля)
        :param volume: каждая координата умножается на volume, по умолчанию volume=1
        """
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x1 = ((np.cos(u) * np.sin(v)).reshape((-1, 1)) + x_offset) * volume
        y1 = ((np.sin(u) * np.sin(v)).reshape((-1, 1)) + y_offset) * volume
        z1 = ((np.cos(v)).reshape((-1, 1)) + z_offset) * volume

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

    @staticmethod
    def get_iris():
        """150 экземплярах ириса, по 50 экземпляров из трёх видов — Ирис щетинистый (Iris setosa),
        Ирис виргинский (Iris virginica) и Ирис разноцветный (Iris versicolor). Для каждого экземпляра измерялись
        четыре характеристики (в см):

        Длина наружной доли околоцветника (англ. sepal length);
        Ширина наружной доли околоцветника (англ. sepal width);
        Длина внутренней доли околоцветника (англ. petal length);
        Ширина внутренней доли околоцветника (англ. petal width).

        Один из классов (Iris setosa) линейно-разделим от двух остальных. """

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        return X, y

    @staticmethod
    def get_digits():
        """Кластеризация набора данных по рукописным цифрам. Картинки здесь представляются матрицей 8 x 8 (интенсивности
         белого цвета для каждого пикселя). Далее эта матрица "разворачивается" в вектор длины 64, получается признаковое
         описание объекта. Размерность признакового пространства здесь – 64. Но если снизим размерность всего до 2 то
         увидим, что даже на глаз рукописные цифры неплохо разделяются на кластеры.
         На практике, как правило, выбирают столько главных компонент, чтобы оставить 90% дисперсии исходных данных.
         В данном случае для этого достаточно выделить 21 главную компоненту, то есть снизить размерность с 64 признаков
         до 21."""

        digits = datasets.load_digits()
        data = digits.data
        X_reduced = data
        pca = decomposition.PCA(n_components=21)
        X_reduced = pca.fit_transform(data)
        return X_reduced, digits.target

    @staticmethod
    def get_web_users():
        """Кластеризация данных о пользователях веб-сайта. Возьмем для примера данные пользователей, которые заходили на
         сайт компании и кликали на ссылки. При помощи подходящего link tracker посчитаем количество кликов по тем или
         иным элементам сайта за определенное время от каждого зарегистрированного пользователя, просуммируем, например,
        по неделям и пронормируем на среднее.

        Начнем с генерации сета, имитирующего поведение трех групп пользователей, кликающих по 5 ссылкам с 1000 разных
        аккаунтов. Для этого воспользуемся методом make_blobs из пакета sklearn.
        В массив data будут записаны пять фич, каждая из которых является суммой трех гауссиан с центрами в трех разных
        точках, которые и являются кластерами. Имена этих заранее известных кластеров записаны в переменной
        pregenerated."""

        data, target = make_blobs(1000, n_features=4)
        return data, target

    @staticmethod
    def compare_results(nn_clusters, real_clusters):
        # Пытаемся привести real_clusters из вида
        # [0, 0, 0, 1, 1]
        # к виду nn_clusters
        # [[0, 1, 2], [3, 4]]
        real_clusters_2 = {}
        for i in range(len(real_clusters)):
            cluster = real_clusters[i]
            if cluster not in real_clusters_2.keys():
                real_clusters_2[cluster] = [i]
            else:
                real_clusters_2[cluster].append(i)
        real_clusters_2 = list(real_clusters_2.values())

        errors = 0
        # Если элемент из кластера real_clusters_2 не принадлежит тому же кластеру в n_clusters, то +ошибка
        for i in range(len(real_clusters_2)):
            for j in range(len(real_clusters_2[i])):
                if i >= len(nn_clusters):
                    errors += 1
                    continue
                elif real_clusters_2[i][j] not in nn_clusters[i]:
                    errors += 1

        return (len(real_clusters) - errors) / len(real_clusters)