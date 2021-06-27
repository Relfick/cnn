from cnn import network
from test_data import test_data
import numpy as np


if __name__ == "__main__":

    add_data = test_data.get_add_sphere()
    add_data_2 = test_data.get_add_sphere_2()
    add_data_3 = test_data.get_add_sphere_3()

    temp = add_data[60:80].copy()
    add_data[50:80] = add_data_2[50:80]
    add_data_2[50:80] = temp

    net = network(num_fluctuations=400)

    y = net.forward(add_data)
    net.visualize_sync()
    # net.visualize_w()
    clusters = net.get_clusters()
    # net.visualize_clusters_3d_2(net.data, clusters)

    y = net.forward(add_data_2)
    net.visualize_sync()
    # net.visualize_w()
    clusters = net.get_clusters()
    print("Num of clusters =", len(clusters))
    # net.visualize_clusters_2d(net.data, clusters)
    # net.visualize_clusters_3d_1(net.data, clusters)
    # net.visualize_clusters_3d_2(net.data, clusters)

    y = net.forward(add_data_3)
    clusters = net.get_clusters()
    print("Num of clusters =", len(clusters))
    net.visualize_sync()
    net.visualize_w()
    net.visualize_clusters_2d()
    net.visualize_clusters_3d_1()
    net.visualize_clusters_3d_2()
