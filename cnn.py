import random
import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import seaborn as sns
from annoy import AnnoyIndex


class network:
    def __init__(self, eps=0.25, tol=0.15, pruning=3, thinning=10, num_fluctuations=100, meeting_period=11):
        """
        :param eps: Допустимая невязка
        :param tol: Толерантность к ошибкам
        :param pruning: Степень обрезки (сколько знаков после запятой)
        :param thinning: Степень прореживания (как часто отбрасываем элементы)
        :param meeting_period: Период "знакомства" нейронов, в анализе его не учитываем
        """
        self.data = np.array([])
        self.y = np.array([])
        self.W = np.array([])
        self.eps = eps
        self.tol = tol
        self.pruning = pruning
        self.thinning = thinning
        self.meeting_period = meeting_period
        self.num_neurons = 0
        self.num_fluctuations = num_fluctuations
        self.old_fluctuations = num_fluctuations
        self.a = 0
        self.colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']
        self.clusters = []

    def forward(self, data):
        if self.data.size > 0:  # clustering on-the-fly
            self.data = np.row_stack((self.data, data))
        else:  # first input
            self.data = data

        new_num_neurons = len(data)
        self.num_neurons = self.num_neurons + len(data)

        self.W = self.calc_w()

        y = np.zeros((self.num_fluctuations + self.meeting_period, self.num_neurons))
        if self.y.size == 0:
            y[0] = np.random.sample(self.num_neurons)
        else:
            old_num_neurons = self.num_neurons - new_num_neurons
            y[0][:old_num_neurons] = self.y.T[-1]
            y[0][old_num_neurons:] = np.random.sample(new_num_neurons)
        self.y = y

        self.make_fluctuations()

        return self.y

    def calc_w(self):
        """ Заполнение W """
        W = np.zeros((self.num_neurons, self.num_neurons))

        if self.a == 0:
            self.a = self.compute_a()

        if self.W.size == 0:
            start_index = 0
        else:
            start_index = self.W.shape[0]

        for i in range(start_index, self.num_neurons):
            for j in range(0, i):
                d = np.linalg.norm(self.data[i] - self.data[j]) ** 2
                W[i][j] = np.exp(-d / (2 * self.a))
            if i % 100 == 0:
                print(f'i = {i}')

        W = W + W.T
        if self.W.size == 0:
            return W
        else:
            W[:self.W.shape[0], :self.W.shape[1]] = self.W
            return W

    def make_fluctuations(self):
        result = self.y
        for t in range(1, self.num_fluctuations + self.meeting_period):
            y_new = self.W @ self.chaos_func(result[t - 1])
            c = 1 / np.sum(self.W, axis=1)
            result[t] = np.diag(c) @ y_new

            result[t] = self.prune(result[t])  # Обрезка

        result = (np.delete(result, range(self.meeting_period), 0)).T  # Удаление рандомных стартовых значений
        result = self.thinning_out(result)  # Прореживание

        self.y = result

    def compute_a_2(self):
        """ Вычисление коэффициента А с помощью триангуляции Делоне """
        a = 0
        tri = Delaunay(self.data)
        print('tri done')
        for i in range(self.num_neurons):
            neighbours = self.get_neighbours(i, tri)
            neighbours = [self.dist(self.data[i], self.data[j]) for j in neighbours]
            if len(neighbours) != 0:
                a += np.array(neighbours).mean()

        a /= self.num_neurons
        return a

    def compute_a(self):
        """ Вычисление коээфициента А с помощью Annoy """
        a = 0
        f = self.data.shape[1]  # Количество параметров примера
        t = AnnoyIndex(f, 'angular')
        for i in range(self.num_neurons):
            t.add_item(i, self.data[i])

        num_trees = 100  # Гиперпараметр, больше деревьев - точнее, но медленнее
        t.build(num_trees)
        t.save('test.ann')

        u = AnnoyIndex(f, 'angular')
        u.load('test.ann')

        if self.num_neurons > 1000:
            num_neighbours = 50  # Гиперпараметр, больше соседей - точнее, но медленнее
        else:
            num_neighbours = round(self.num_neurons / 10)
        for i in range(self.num_neurons):
            neighbours = u.get_nns_by_item(i, num_neighbours)  # Ищем по num_neighbours соседей для каждой точки
            neighbours = [self.dist(self.data[i], self.data[j]) for j in neighbours]
            if len(neighbours) != 0:
                a += np.array(neighbours).mean()

        a /= self.num_neurons
        print(f'a = {a}')
        return a

    def dist(self, x, y):
        return (((x - y) ** 2).sum()) ** 0.5

    def get_neighbours(self, k, tri):
        """ Для триангуляции """
        indptr, indices = tri.vertex_neighbor_vertices
        indices = indices[indptr[k]:indptr[k + 1]]
        return indices

    def chaos_func(self, y):
        ret = 1 - 2 * (y ** 2)
        return ret

    def prune(self, y):
        """ Обрезка """
        if self.pruning:
            return np.round(y, self.pruning)

    def thinning_out(self, y, discard=False):
        """
        Реализует прореживание, если self.thinning != False
        :param y:
        :param discard: По умолчанию False - реализует сохранение каждого self.thinning колебания,
        остальные отбрасываются. Если True - отбрасывается каждое self.thinning колебание, остальные сохраняются.
        """
        y_thinned = y
        if self.thinning:
            if discard:
                y_thinned = np.delete(y, slice(self.thinning - 1, None, self.thinning), axis=1)
            else:
                y_thinned = y[:, self.thinning - 1::self.thinning]
            self.num_fluctuations = len(y_thinned[0])
        return y_thinned

    def get_clusters(self):
        self.clusters = [list(i) for i in self.form_clusters()]
        self.num_fluctuations = self.old_fluctuations
        return self.clusters

    def form_clusters(self):
        P = self.calc_p()
        P_dop = P < self.tol

        clusters = []
        for i in range(self.num_neurons):
            i_clusters = set()
            for j in range(self.num_neurons):
                if P_dop[i][j]:
                    i_clusters.add(j)
            if i_clusters not in clusters:
                clusters.append(i_clusters)
        return self.condense_sets(clusters)

    def calc_g(self):
        """ Бинарный вектор G """
        M = self.calc_m()
        G = (M <= self.eps).astype(int)
        return G

    def calc_p(self):
        """ Процент отсчетов, в которых нарушается граница, установленная допустимой невязкой """
        G = self.calc_g()
        P = np.zeros((self.num_neurons, self.num_neurons))
        for i in range(self.num_neurons):
            P[i] = 1 - np.count_nonzero(G[i], axis=1) / self.num_fluctuations
        return P

    def calc_m(self):
        """ Матрицы невязок M """
        M = np.zeros((self.num_neurons, self.num_neurons, self.num_fluctuations))
        for i in range(self.num_neurons):
            for k in range(self.num_neurons):
                M[i][k] = np.abs(self.y[k] - self.y[i])
        return M

    def condense_sets(self, sets):
        """ Объединяет пересекающиеся сеты в один сет """
        result = []
        for candidate in sets:
            for current in result:
                if candidate & current:  # found overlap
                    current |= candidate  # combine (merge sets)

                    # new items from candidate may create an overlap
                    # between current set and the remaining result sets
                    result = self.condense_sets(result)  # merge such sets
                    break
            else:  # no common elements found (or result is empty)
                result.append(candidate)

        self.colors = self.colors + ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                                     for i in range(len(result) - 7)]

        return result

    def visualize_sync(self):
        fig = plt.figure()
        ax = fig.add_subplot()

        data = np.around(self.y, 2)

        sns.heatmap(data, ax=ax, cmap="rainbow")

        fig.set_figwidth(14)
        fig.set_figheight(6)

        plt.show()

    def visualize_w(self):
        fig = plt.figure()
        ax = fig.add_subplot()

        data = self.W

        sns.heatmap(data, ax=ax, cmap="rainbow")

        fig.set_figwidth(14)
        fig.set_figheight(6)

        plt.show()

    def visualize_data(self):
        fig = plt.figure()
        ax = fig.add_subplot()

        data = self.data

        sns.heatmap(data, ax=ax, cmap="rainbow")

        fig.set_figwidth(14)
        fig.set_figheight(6)

        plt.show()

    def visualize_clusters_3d_1(self):
        ax = plt.axes(projection='3d')

        for i in range(len(self.clusters)):
            cluster_points = self.data[self.clusters[i]]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], marker='o', color=self.colors[i])

        plt.show()

    def visualize_clusters_3d_2(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(self.clusters)):
            cluster_points = self.data[self.clusters[i]]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=self.colors[i], marker='x')

        plt.show()

    def visualize_clusters_2d(self):
        for i in range(len(self.clusters)):
            cluster_points = self.data[self.clusters[i]]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], marker='o', color=self.colors[i])

        plt.show()
