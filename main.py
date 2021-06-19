import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.datasets

# Исходные данные

# Изображение
# MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
#
# X_train = MNIST_train.data
# y_train = MNIST_train.targets
#
# X_train = X_train.float()
#
# plt.imshow(X_train[4, :, :])
# plt.show()
# print(y_train[4])
#
# X_train = X_train.reshape([-1, 10*10])
#
#
# x = X_train.numpy()[:150, :]


# Две сферы
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
x = np.concatenate([x, y, z], axis=1)

# Random
# x = np.array([[0., 0.5], [1., 1.], [-1., 2.], [-1., -1.], [2., 1.]])
# x = np.random.uniform(-28, 28, (140, 2))

# Количество объектов
length = len(x)

# Время наблюдения
times = 100


# Предобработка

# Визуализация триангуляции
def visual_tri(tri, x):
    # если x - координаты на плоскости, можно посмотреть визуализацию
    plt.triplot(x[:, 0], x[:, 1], tri.simplices)
    plt.plot(x[:, 0], x[:, 1], 'o')
    plt.show()


def dist(x, y):
    return (((x - y) ** 2).sum()) ** 0.5


def get_neighbours(k, tri):
    indptr, indices = tri.vertex_neighbor_vertices
    indices = indices[indptr[k]:indptr[k + 1]]
    return indices


# Расчет масштабирующей константы на основе построения триангуляции Делоне
def compute_a(x):
    a = 0
    tri = Delaunay(x)

    for i in range(length):
        neighbours = get_neighbours(i, tri)
        neighbours = [dist(x[i], x[j]) for j in neighbours]
        if len(neighbours) != 0:
            a += np.array(neighbours).mean()

    a /= len(x)
    return a


# Вычисление весовых коэффициентов
def calc_w():
    """Заполнение W"""
    W = np.zeros((length, length))
    a = compute_a(x)

    for i in range(length):
        for j in range(i + 1, length):
            d = np.linalg.norm(x[i] - x[j]) ** 2
            W[i][j] = np.exp(-d / (2 * a))

    return W + W.T


# Основная часть
def func(y):
    return 1 - 2 * (y ** 2)


def forward(W):
    result = np.zeros((times + 1, length))
    result[0] = np.random.sample(length)
    for t in range(1, times + 1):
        y_new = W @ func(result[t - 1])
        c = 1 / np.sum(W, axis=1)
        result[t] = np.diag(c) @ y_new

        # Обрезка
        if pruningON:
            result[t] = np.round(result[t], pruning)

    print("y =", result)
    return np.delete(result, 0, 0)


# Постобработка

# Прореживание
def thinning_out(y):
    y_thinned = np.delete(y, slice(thinning - 1, None, thinning), axis=1)
    global times, length
    times = len(y_thinned[0])
    length = len(y_thinned)
    print(len(y_thinned))
    print(len(y_thinned[0]))
    return y_thinned


# Матрицы невязок M
def calc_m(y):
    M = np.zeros((length, length, times))
    dy = y.T
    for i in range(length):
        for k in range(length):
            M[i][k] = np.abs(dy[k] - dy[i])
    print("M =", M)
    return M


# Бинарный вектор G
def calc_g(y, eps):
    y1 = y.T
    M = calc_m(y1)
    G = (M <= eps).astype(int)
    print("G =", G)
    return G


# Процент отсчетов, в которых нарушается граница, установленная допустимой невязкой
def calc_p(y):
    G = calc_g(y, eps)
    P = np.zeros((length, length))
    for i in range(length):
        P[i] = 1 - np.count_nonzero(G[i], axis=1) / times
    print("P =", P)
    return P


# Формирование кластеров
def form_clusters(y):
    P = calc_p(y)
    P_dop = P < tol

    clusters = []
    for i in range(length):
        i_clusters = set()
        for j in range(length):
            if P_dop[i][j]:
                i_clusters.add(j)
        if i_clusters not in clusters:
            clusters.append(i_clusters)
    return condense_sets(clusters)


# Объединяет пересекающиеся сеты в один сет
def condense_sets(sets):
    result = []
    for candidate in sets:
        for current in result:
            if candidate & current:  # found overlap
                current |= candidate  # combine (merge sets)

                # new items from candidate may create an overlap
                # between current set and the remaining result sets
                result = condense_sets(result)  # merge such sets
                break
        else:  # no common elements found (or result is empty)
            result.append(candidate)
    return result


# Визуализация фрагментарной синхронизации
def visualize(y):
    fig = plt.figure()
    ax = fig.add_subplot()

    data = np.around(y, 2)

    sns.heatmap(data, ax=ax, cmap="rainbow")

    fig.set_figwidth(14)
    fig.set_figheight(6)

    plt.show()


# Визуализация кластеров (2D)
def visualize_clusters_2d(x, clusters):
    # can only paint in 7 colors
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']

    for i in range(len(clusters)):
        cluster_points = x[clusters[i]]
        plt.plot(cluster_points[:, 0], cluster_points[:, 1], colors[i] + 'o')

    plt.show()


# Визуализация кластеров (3D - 1й вид)
def visualize_clusters_3d_1(x, clusters):
    # can only paint in 7 colors
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']

    ax = plt.axes(projection='3d')

    for i in range(len(clusters)):
        cluster_points = x[clusters[i]]
        plt.plot(cluster_points[:, 0], cluster_points[:, 1], colors[i] + 'o')

    plt.show()


# Визуализация кластеров (3D - 2й вид)
def visualize_clusters_3d_2(x, clusters):
    # can only paint in 7 colors
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(clusters)):
        cluster_points = x[clusters[i]]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=colors[i], marker='x')

    plt.show()


# Визуализация аттракторов (фазовый портрет)
def visuzlize_dyn():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(y[:, 0].reshape(10, 10), y[:, 1].reshape(10, 10), y[:, 2].reshape(10, 10))

    plt.plot(y[:, 0], y[:, 1])
    plt.plot(y[:, 1], y[:, 2])
    plt.plot(y[:, 0], y[:, 2])


if __name__ == "__main__":
    # Допустимая невязка
    eps = 0.25

    # Толерантность к ошибкам
    tol = 0.15

    # Степень обрезки (сколько знаков после запятой)
    pruningON = True
    pruning = 3

    # Степень прореживания (как часто отбрасываем элементы)
    thinningON = True
    thinning = 10

    # Start
    print("x =", x)
    y = forward(calc_w()).T

    if thinningON:
        y = thinning_out(y)

    visualize(y)

    clusters = [list(i) for i in form_clusters(y)]
    print("clusters =", clusters)

    visualize_clusters_2d(x, clusters)
    visualize_clusters_3d_2(x, clusters)
