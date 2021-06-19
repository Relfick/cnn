import numpy as np
from cnn import network
from test_data import test_data

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


if __name__ == "__main__":

    data = test_data.get_spheres()
    net = network()

    num_fluctuations = 100

    y = net.forward(data, num_fluctuations)

    net.visualize_sync()
    clusters = net.get_clusters()

    net.visualize_clusters_2d(data, clusters)
    net.visualize_clusters_3d_2(data, clusters)
