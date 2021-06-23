from cnn import network
from test_data import test_data


if __name__ == "__main__":

    data = test_data.get_web_users()
    net = network()

    num_fluctuations = 100

    y = net.forward(data, num_fluctuations)

    net.visualize_sync()
    clusters = net.get_clusters()

    net.visualize_clusters_2d(data, clusters)
    net.visualize_clusters_3d_2(data, clusters)
