import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class network:
    def __init__(self, eps=0.25, tol=0.15, pruning=3, thinning=10):
        """
        :param eps: Допустимая невязка
        :param tol: Толерантность к ошибкам
        :param pruning: Степень обрезки (сколько знаков после запятой)
        :param thinning: Степень прореживания (как часто отбрасываем элементы)
        """
        self.data = []
        self.y = []
        self.W = []
        self.eps = eps
        self.tol = tol
        self.pruning = pruning
        self.thinning = thinning
        self.num_neurons = 0
        self.num_fluctuations = 0

    def forward(self, data, num_fluctuations):
        self.num_fluctuations = num_fluctuations
        self.data = data
        self.num_neurons = len(data)

        self.W = self.calc_w()

        result = np.zeros((self.num_fluctuations + 1, self.num_neurons))
        result[0] = np.random.sample(self.num_neurons)
        for t in range(1, self.num_fluctuations + 1):
            y_new = self.W @ self.chaos_func(result[t - 1])
            c = 1 / np.sum(self.W, axis=1)
            result[t] = np.diag(c) @ y_new

            # Обрезка
            if self.pruning:
                result[t] = np.round(result[t], self.pruning)

        result = (np.delete(result, 0, 0)).T
        # Прореживание
        if self.thinning:
            result = self.thinning_out(result)

        self.y = result
        return result

    def calc_w(self):
        """Заполнение W"""
        W = np.zeros((self.num_neurons, self.num_neurons))
        a = self.compute_a()

        for i in range(self.num_neurons):
            for j in range(i + 1, self.num_neurons):
                d = np.linalg.norm(self.data[i] - self.data[j]) ** 2
                W[i][j] = np.exp(-d / (2 * a))

        return W + W.T

    def compute_a(self):
        a = 0
        tri = Delaunay(self.data)

        for i in range(self.num_neurons):
            neighbours = self.get_neighbours(i, tri)
            neighbours = [self.dist(self.data[i], self.data[j]) for j in neighbours]
            if len(neighbours) != 0:
                a += np.array(neighbours).mean()

        a /= self.num_neurons
        return a

    def dist(self, x, y):
        return (((x - y) ** 2).sum()) ** 0.5

    def get_neighbours(self, k, tri):
        indptr, indices = tri.vertex_neighbor_vertices
        indices = indices[indptr[k]:indptr[k + 1]]
        return indices

    def chaos_func(self, y):
        return 1 - 2 * (y ** 2)

    def thinning_out(self, y):
        y_thinned = np.delete(y, slice(self.thinning - 1, None, self.thinning), axis=1)
        self.num_fluctuations = len(y_thinned[0])
        self.num_neurons = len(y_thinned)
        return y_thinned

    def get_clusters(self):
        return [list(i) for i in self.form_clusters()]

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
        return result

    def visualize_sync(self):
        fig = plt.figure()
        ax = fig.add_subplot()

        data = np.around(self.y, 2)

        sns.heatmap(data, ax=ax, cmap="rainbow")

        fig.set_figwidth(14)
        fig.set_figheight(6)

        plt.show()

    def visualize_clusters_3d_1(self, x, clusters):
        # can only paint in 7 colors
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']

        ax = plt.axes(projection='3d')

        for i in range(len(clusters)):
            cluster_points = x[clusters[i]]
            plt.plot(cluster_points[:, 0], cluster_points[:, 1], colors[i] + 'o')

        plt.show()

    def visualize_clusters_3d_2(self, x, clusters):
        # can only paint in 7 colors
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(clusters)):
            cluster_points = x[clusters[i]]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=colors[i], marker='x')

        plt.show()

    def visualize_clusters_2d(self, x, clusters):
        # can only paint in 7 colors
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']

        for i in range(len(clusters)):
            cluster_points = x[clusters[i]]
            plt.plot(cluster_points[:, 0], cluster_points[:, 1], colors[i] + 'o')

        plt.show()