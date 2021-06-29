from cnn import network
from test_data import test_data

if __name__ == "__main__":
    data_spheres = test_data.get_spheres()
    data_add_sphere_1 = test_data.get_add_sphere()
    data_add_sphere_2 = test_data.get_add_sphere_2()
    data_add_sphere_3 = test_data.get_add_sphere_3()
    data_mnist, target_mnist = test_data.get_mnist()
    data_iris = test_data.get_iris()
    data_web_users = test_data.get_web_users()
    data_digits = test_data.get_digits()

    data = data_digits
    data = test_data.normalize_minmax(data)
    # data = test_data.normalize_minmax(data)
    # data = np.row_stack((data, data_add_sphere_1))
    # data = np.row_stack((data, data_add_sphere_2))
    # data = np.row_stack((data, data_add_sphere_3))
    test_data.visualize_heatmap(data)

    net = network(num_fluctuations=1000, meeting_period=100)
    net.forward(data)
    net.visualize_w()
    net.visualize_sync()
    clusters = net.get_clusters()
    print(len(clusters))
    net.visualize_clusters_3d_2()
