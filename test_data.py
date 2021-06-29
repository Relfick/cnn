import torchvision.datasets
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, datasets, __all__, decomposition
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

    @staticmethod
    def get_iris():
        """Кластеризация Ирисов Фишера. Это набор данных для задачи классификации, на примере которого Рональд Фишер в
        1936 году продемонстрировал работу разработанного им метода дискриминантного анализа. Иногда его также называют
        ирисами Андерсона, так как данные были собраны американским ботаником Эдгаром Андерсоном.
        Ирисы Фишера состоят из данных о 150 экземплярах ириса, по 50 экземпляров из трёх видов — Ирис щетинистый (Iris setosa),
        Ирис виргинский (Iris virginica) и Ирис разноцветный (Iris versicolor). Для каждого экземпляра измерялись
        четыре характеристики (в сантиметрах):

        Длина наружной доли околоцветника (англ. sepal length);
        Ширина наружной доли околоцветника (англ. sepal width);
        Длина внутренней доли околоцветника (англ. petal length);
        Ширина внутренней доли околоцветника (англ. petal width).

        На основании этого набора данных требуется построить правило классификации, определяющее вид растения по данным
        измерений. Это задача многоклассовой классификации, так как имеется три класса — три вида ириса.

        Один из классов (Iris setosa) линейно-разделим от двух остальных. """

        iris_df = datasets.load_iris()
        X_train = iris_df.data

        return X_train

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
        pca = decomposition.PCA(n_components=21)
        X_reduced = pca.fit_transform(data)

        return X_reduced

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

        data, pregenerated = make_blobs(1000, n_features=5, cluster_std=4)
        return data
