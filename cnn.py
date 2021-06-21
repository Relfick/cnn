import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import seaborn as sns


class network:
    def __init__(self, eps=0.25, tol=0.15, pruning=3, thinning=10, num_fluctuations=100):
        """
        :param eps: Допустимая невязка
        :param tol: Толерантность к ошибкам
        :param pruning: Степень обрезки (сколько знаков после запятой)
        :param thinning: Степень прореживания (как часто отбрасываем элементы)
        """
        self.data = np.array([])
        self.y = np.array([])
        self.W = np.array([])
        self.eps = eps
        self.tol = tol
        self.pruning = pruning
        self.thinning = thinning
        self.num_neurons = 0
        self.num_fluctuations = num_fluctuations
        self.a = 0

    def forward(self, data):
        """ пока в дате всё """
        if self.data.size > 0:  # clustering on-the-fly
            self.data = np.row_stack((self.data, data))

            new_num_neurons = len(data)
            self.num_neurons = self.num_neurons + len(data)

            self.W = self.calc_w()
        else:  # first input
            self.data = data

            new_num_neurons = len(data)
            self.num_neurons = len(data)

            self.W = self.calc_w()

        result = self.make_fluctuations(new_num_neurons)

        self.y = result
        return result

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
            for j in range(i + 1, self.num_neurons):
                d = np.linalg.norm(self.data[i] - self.data[j]) ** 2
                W[i][j] = np.exp(-d / (2 * self.a))

        W = W + W.T
        if self.W.size == 0:
            return W
        else:
            W[:self.W.shape[0], :self.W.shape[1]] = self.W
            return W

    def make_fluctuations(self, new_num_neurons):

        if self.y.size == 0:
            result = np.zeros((self.num_fluctuations + 1, self.num_neurons))
            result[0] = np.random.sample(self.num_neurons)
            for t in range(1, self.num_fluctuations + 1):
                y_new = self.W @ self.chaos_func(result[t - 1])
                c = 1 / np.sum(self.W, axis=1)
                result[t] = np.diag(c) @ y_new

                if self.pruning:  # Обрезка
                    result[t] = np.round(result[t], self.pruning)

            result = (np.delete(result, 0, 0)).T  # Удаление рандомных стартовых значений

            # if self.thinning:  # Прореживание
            #     result = self.thinning_out(result)
        else:
            W = self.W[-new_num_neurons:]
            result = np.zeros((self.num_fluctuations + 1, new_num_neurons))
            result[0] = np.random.sample(new_num_neurons)
            self.y = np.row_stack((np.random.sample(self.y.shape[0]), self.y.T))  # т.к. в у уже нет рандомного слоя
            result = np.column_stack((self.y, result))
            for t in range(1, self.num_fluctuations + 1):
                y_new = W @ self.chaos_func(result[t - 1])  # (200, 600) x (600, 1)
                c = 1 / np.sum(W, axis=1)
                result[t][-new_num_neurons:] = np.diag(c) @ y_new

                if self.pruning:  # Обрезка
                    result[t][-new_num_neurons:] = np.round(result[t][-new_num_neurons:], self.pruning)

            result = (np.delete(result, 0, 0)).T

            # if self.thinning:  # Прореживание
            #     result = self.thinning_out(result)

        return result


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
        ret = 1 - 2 * (y ** 2)
        return ret

    def thinning_out(self, y):
        # y_thinned = np.delete(y, slice(self.thinning - 1, None, self.thinning), axis=1)
        y_thinned = y[:, self.thinning - 1::self.thinning]
        self.num_fluctuations = len(y_thinned[0])
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

    def visualize_w(self):
        fig = plt.figure()
        ax = fig.add_subplot()

        data = self.W

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