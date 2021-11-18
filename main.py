from cnn import network
from test_data import test_data
import numpy as np

if __name__ == "__main__":
    sphere1 = test_data.get_sphere()
    sphere2 = test_data.get_sphere(volume=10)
    sphere3 = test_data.get_sphere(volume=20)
    data1 = np.row_stack((sphere1, sphere2, sphere3))
    data2, target = test_data.get_iris()
    data3 = test_data.get_digits()
    data4 = test_data.get_web_users()
    data5 = test_data.get_spheres()
    data_add_sphere_1 = test_data.get_add_sphere()
    data_add_sphere_2 = test_data.get_add_sphere_2()
    data_add_sphere_3 = test_data.get_add_sphere_3()
    # data_mnist, target_mnist = test_data.get_mnist()

    # data = data_mnist
    # data = test_data.normalize_minmax(data)
    data = data_add_sphere_1
    data = np.row_stack((data, data_add_sphere_2))
    data = np.row_stack((data, data_add_sphere_3))
    # test_data.visualize_heatmap(data)

    net = network(num_fluctuations=1000, meeting_period=100, tetta=0.5)
    net.forward(data5)
    net.visualize_w()
    net.visualize_sync()
    clusters = net.get_clusters(1)
    print(f'Number of clusters = {len(clusters)}')
    print(f'Correctly clustered: {round(test_data.compare_results(clusters, target) * 100)}%')
    print(clusters)
    net.visualize_clusters_3d_2()
    net.visualize_clusters_2d()
