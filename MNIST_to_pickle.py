import cupy as cp
from tensorflow.keras.datasets import mnist
import pickle

(x_train, temp_train), (x_test, temp_test) = mnist.load_data()

x_train = cp.array(x_train) / 255.0
x_test = cp.array(x_test) / 255.0
temp_train = cp.array(temp_train)
temp_test = cp.array(temp_test)


shuffled_index_train = cp.arange(x_train.shape[0])
cp.random.shuffle(shuffled_index_train)
x_train, temp_train = x_train[shuffled_index_train], temp_train[shuffled_index_train]

shuffled_index_test = cp.arange(x_test.shape[0])
cp.random.shuffle(shuffled_index_test)
x_test, temp_test = x_test[shuffled_index_test], temp_test[shuffled_index_test]


# t -> one-hot encoding
t_train = cp.zeros((60000, 10))
t_test = cp.zeros((10000, 10))
for i, n in enumerate(temp_train):
    t_train[i][n] = 1
for i, n in enumerate(temp_test):
    t_test[i][n] = 1


# original 2D dataset
with open("MNIST_onehot.pickle", "wb") as fw:
    pickle.dump(x_train, fw)
    pickle.dump(x_test, fw)
    pickle.dump(t_train, fw)
    pickle.dump(t_test, fw)


# Flattened 1D dataset
x_train = cp.array(x_train.reshape(60000, 784) / 255)
x_test = cp.array(x_test.reshape(10000, 784) / 255)

with open("MNIST_flattened_onehot.pickle", "wb") as fw:
    pickle.dump(x_train, fw)
    pickle.dump(x_test, fw)
    pickle.dump(t_train, fw)
    pickle.dump(t_test, fw)
    