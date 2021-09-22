import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DLwithcp import functions, layers, optimizers


class CNN:
    def __init__(self, input_dimension=(1, 28, 28), filters=5, filter_size=(5, 5), \
        hidden_size=100, output_size=10, init_std = 0.1):
        C, H, W = input_dimension
        FN = filters
        FH, FW = filter_size
        OH, OW = H - FH + 1, W - FW + 1
        
        self.parameters = {}
        self.parameters['W1'] = init_std * \
            cp.random.randn(filters, input_dimension[0], filter_size[0], filter_size[1])  # filters
        self.parameters['b1'] = cp.zeros(filters)  # bias in convolution
        self.parameters['W2'] = init_std * \
            cp.random.randn(int(FN*OH*OW / (2**2)), hidden_size)
        self.parameters['b2'] = cp.zeros(hidden_size)
        self.parameters['W3'] = init_std * \
            cp.random.randn(hidden_size, output_size)
        self.parameters['b3'] = cp.zeros(output_size)

        self.layers = {}
        self.layers['Conv1'] = layers.Convolution(self.parameters['W1'], self.parameters['b1'])
        self.layers['ReLU1'] = layers.ReLU()
        self.layers['Pooling1'] = layers.Pooling((2, 2))
        self.layers['Flatten'] = layers.Flatten()
        self.layers['Affine1'] = layers.Affine(self.parameters['W2'], self.parameters['b2'])
        self.layers['ReLU2'] = layers.ReLU()
        self.layers['Affine2'] = layers.Affine(self.parameters['W3'], self.parameters['b3'])

        self.last_layer = layers.SoftmaxWithCrossEntropy()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = cp.argmax(y, axis=1)
        t = cp.argmax(t, axis=1)
        return cp.sum(y == t) / len(x)

    def back_propagation(self, x, t):
        self.loss_ = self.loss(x, t)  # Some functions need forward before backward
        
        dLdx = 1
        dLdx = self.last_layer.backward(dLdx)
        for layer in reversed(self.layers.values()):
            dLdx = layer.backward(dLdx)
        
        gradients = {}
        gradients['W1'] = self.layers['Conv1'].dLdF
        gradients['b1'] = self.layers['Conv1'].dLdB
        gradients['W2'] = self.layers['Affine1'].dLdW
        gradients['b2'] = self.layers['Affine1'].dLdb
        gradients['W3'] = self.layers['Affine2'].dLdW
        gradients['b3'] = self.layers['Affine2'].dLdb
        return gradients


# Load MNIST dataset
with open("datasets\MNIST_onehot.pickle", "rb") as fr:
    x_train = pickle.load(fr)
    x_test = pickle.load(fr)
    t_train = pickle.load(fr)
    t_test = pickle.load(fr)

x_train = x_train.reshape(60000, 1, 28, 28)
x_test = x_test.reshape(10000, 1, 28, 28)
x_test, t_test = x_test[:1000], t_test[:1000]

# Apply model to MNIST dataset
model = CNN()
optimizer = optimizers.Adam()
batch_size = 100
train_loss_list = []
test_loss_list = []
test_acc_list = []
iter_list = []


for iter_ in tqdm(range(30)):
    batch_mask = cp.random.choice(len(x_train), batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    gradients = model.back_propagation(x_batch, t_batch)
    optimizer.update(model.parameters, gradients)

    if (iter_+1) % 10 == 0:
        y_test = functions.softmax(model.predict(x_test))
        test_loss = functions.cross_entropy(y_test, t_test)
        test_acc = cp.sum(y_test.argmax(axis=1) == t_test.argmax(axis=1)) / y_test.shape[0]

        train_loss_list.append(float(model.loss_))
        test_loss_list.append(float(test_loss))
        test_acc_list.append(float(test_acc))
        iter_list.append(iter_+1)

        tqdm.write(f"iter: {iter_+1} / train_loss: {model.loss_} / test_loss: {test_loss} / test_acc: {test_acc}")

    else:
        tqdm.write(f"iter: {iter_+1} / train_loss: {model.loss_}")


# Plot train/test loss and test accuracy vs iteration
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

train_loss_plot = ax1.plot(iter_list, train_loss_list, label='train_loss')
test_loss_plot = ax1.plot(iter_list, test_loss_list, label='test_loss')
ax1.set_ylabel('loss')
ax1.legend(loc='upper right')

ax2.plot(iter_list, test_acc_list, label='test_acc')
ax2.set_xlabel('iteration')
ax2.set_ylabel('accuracy')
ax2.legend(loc='lower right')
plt.show()
