from cnn import network
from test_data import test_data
import random
import numpy as np
from annoy import AnnoyIndex


if __name__ == "__main__":
    data_spheres = test_data.get_spheres()
    add_data = test_data.get_add_sphere()
    add_data_2 = test_data.get_add_sphere_2()
    add_data_3 = test_data.get_add_sphere_3()
    data_mnist = test_data.get_mnist()

    data, targets = data_mnist
    # data = np.row_stack((data, add_data))
    # data = np.row_stack((data, add_data_2))
    # data = np.row_stack((data, add_data_3))

    net = network(num_fluctuations=1000, meeting_period=100)
    net.forward(data)
    net.visualize_sync()
    # net.visualize_w()
    clusters = net.get_clusters()
    print(len(clusters))
    # net.visualize_clusters_3d_2()
