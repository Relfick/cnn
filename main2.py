from cnn import network
from test_data import test_data


if __name__ == "__main__":

    data = test_data.get_spheres()
    num_fluctuations = 100

    net = network(num_fluctuations)

    y = net.forward(data)

    net.visualize_sync()
    clusters = net.get_clusters()

    net.visualize_clusters_2d(data, clusters)
    net.visualize_clusters_3d_2(data, clusters)
