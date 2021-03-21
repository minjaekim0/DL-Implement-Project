import cupy as cp
from tensorflow.keras.datasets import mnist
import pickle

(x_train, temp_train), (x_test, temp_test) = mnist.load_data()
x_train = cp.array(x_train.reshape(60000, 784) / 255)
x_test = cp.array(x_test.reshape(10000, 784) / 255)

# t -> one-hot encoding
t_train = cp.zeros((60000, 10))
t_test = cp.zeros((10000, 10))
for i, n in enumerate(temp_train):
    t_train[i][n] = 1
for i, n in enumerate(temp_test):
    t_test[i][n] = 1

with open("MNIST_onehot.pickle", "wb") as fw:
    pickle.dump(x_train, fw)
    pickle.dump(x_test, fw)
    pickle.dump(t_train, fw)
    pickle.dump(t_test, fw)
    