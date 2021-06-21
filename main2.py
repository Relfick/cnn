from cnn import network
from test_data import test_data
import numpy as np


if __name__ == "__main__":

    data = test_data.get_spheres()
    add_data = test_data.get_add_sphere()
    add_data_2 = test_data.get_add_sphere_2()
    add_data_3 = test_data.get_add_sphere_3()

    add_data[:30], add_data_2[:30] = add_data_2[:30], add_data[:30]

    # net = network()

    # data = np.row_stack((data, add_data))
    # data = np.row_stack((data, add_data_2))
    data = np.row_stack((data, add_data_3))

    eps, tol = 0.25, 0.15
    while tol < 0.8:
        while eps < 0.9:
            net = network(eps=eps, tol=tol)
            y = net.forward(data)
            clusters = net.get_clusters()
            print(f"eps={eps}, tol={tol}")
            print(f"clusters:{len(clusters)}")
            if len(clusters) == 2:
                break

            eps += 0.05
        tol += 0.05
        eps = 0.25


    # y = net.forward(data)
    # net.visualize_sync()
    # clusters = net.get_clusters()
    # net.visualize_clusters_3d_2(data, clusters)

    # y = net.forward(add_data)
    # net.visualize_sync()
    # clusters = net.get_clusters()
    # net.visualize_clusters_3d_2(add_data, clusters)
    #
    # y = net.forward(add_data_2)
    # net.visualize_sync()
    # clusters = net.get_clusters()
    # net.visualize_clusters_3d_2(np.row_stack((add_data, add_data_2)), clusters)
    #
    # y = net.forward(add_data_3)
    # net.visualize_sync()
    # clusters = net.get_clusters()
    # net.visualize_clusters_3d_2(np.row_stack((np.row_stack((add_data, add_data_2)), add_data_3)), clusters)
    #
    # y = net.forward(data)
    # net.visualize_sync()
    # clusters = net.get_clusters()
    #
    # data = np.row_stack((data, add_data))
    # data = np.row_stack((data, add_data_2))
    # data = np.row_stack((data, add_data_3))
    #
    # net.visualize_clusters_2d(data, clusters)
    # net.visualize_clusters_3d_2(data, clusters)
