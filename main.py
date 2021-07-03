import numpy as np

from cnn import network
from test_data import test_data

if __name__ == "__main__":
    sphere1 = test_data.get_sphere()
    sphere2 = test_data.get_sphere(volume=5)

    data = np.row_stack((sphere1, sphere2))

    net = network()
    net.forward(data)
    net.visualize_w()
    print(len(net.get_clusters()))
    net.visualize_clusters_3d_2()