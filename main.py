import numpy as np
from matplotlib import pyplot as plt

from cnn import network
from test_data import test_data

if __name__ == "__main__":
    """ Spheres with offsets """
    # sphere1 = test_data.get_sphere()
    # sphere2 = test_data.get_sphere(x_offset=3)
    # sphere3 = test_data.get_sphere(z_offset=5)
    # data = np.row_stack((sphere1, sphere2, sphere3))
    # target = [[i]*200 for i in range(data.shape[1])]
    # target = sum(target, [])

    """ Spheres one in another """
    sphere1 = test_data.get_sphere()
    sphere2 = test_data.get_sphere(volume=3)
    sphere3 = test_data.get_sphere(volume=5)
    data = np.row_stack((sphere1, sphere2, sphere3))
    target = [[i]*200 for i in range(data.shape[1])]
    target = sum(target, [])

    """ Iris """
    # data, target = test_data.get_iris()

    """ Digits """
    # data, target = test_data.get_digits()
    # plt.imshow(data[2].reshape((8,8)))
    # plt.show()
    # data = test_data.normalize_minmax(data)

    """ MNIST """
    # data, target = test_data.get_mnist()
    # plt.imshow(data[0].reshape((28, 28)))
    # plt.show()
    # data = test_data.normalize_minmax(data)


    net = network()
    net.forward(data, compute_a_method='del')
    net.visualize_w()
    net.visualize_sync()

    clusters = net.get_clusters()
    print(f'Number of clusters = {len(clusters)}')
    print(f'Correctly clustered: {round(test_data.compare_results(clusters, target) * 100)}%')
    net.visualize_clusters_2d()
    net.visualize_clusters_3d_2()

    """ Accuracy """
    # net = network()
    # best_eps, best_tol, best_correctly_clustered = \
    #     net.smart_clustering_accuracy(data, target, compute_a_method='del')
    # print(f'Best eps = {best_eps}, best tol = {best_tol}')
    # print(f'Best correctly clustered = {best_correctly_clustered * 100}%')
    # net.visualize_clusters_3d_2()
    # net.visualize_clusters_2d()

    """ Frequency """
    # net = network()
    # freq_eps, freq_tol, max_freq, correctly_clustered = \
    #     net.smart_clustering_frequency(data, target, compute_a_method='del')
    # print(f'eps = {freq_eps}, tol = {freq_tol}')
    # print(f'Frequent = {max_freq}')
    # print(f'Correctly clustered = {correctly_clustered * 100}%')
    # net.visualize_clusters_3d_2()
    # net.visualize_clusters_2d()