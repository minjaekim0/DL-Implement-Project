import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy as cp
import pickle


class Affine: # A = xW + b
    def __init__(self, W, b):
        self.W = W
        self.b = b
    
    def forward(self, x):
        self.x = x
        return cp.dot(x, self.W) + self.b
    
    def backward(self, dLdA):
        self.dLdW = cp.dot(self.x.T, dLdA)
        self.dLdb = cp.sum(dLdA, axis=0)
        dLdx = cp.dot(dLdA, self.W.T)
        return dLdx


class Relu: # y = Relu(x)
    def forward(self, x):
        self.x = x
        x_ = x.copy()
        x_[x_ < 0] = 0
        return x_
    
    def backward(self, dLdy):
        dLdx = dLdy.copy()
        dLdx[self.x < 0] = 0
        return dLdx


def softmax(x):
    x = x - cp.max(x, axis=1).reshape(-1, 1)
    return cp.exp(x) / cp.sum(cp.exp(x), axis=1).reshape(-1, 1)

def CE(y, t):
    if y.ndim == 1:
        return -cp.sum(t * cp.log(y + 1e-5))
    else:
        return -cp.sum(t * cp.log(y + 1e-5)) / len(y)

class Softmax_with_CE: # y = softmax(x), L = CE(y, t)
    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        self.L = CE(self.y, self.t)
        return self.L
    
    def backward(self, dLdL=1):
        if self.t.ndim == 1:
            dLdx = self.y - self.t
        else:
            dLdx = (self.y - self.t) / len(self.t)
        return dLdx


class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.parameters = {}
        self.parameters['W1'] = cp.random.randn(input_size, hidden_size)
        self.parameters['b1'] = cp.zeros(hidden_size)
        self.parameters['W2'] = cp.random.randn(hidden_size, output_size)
        self.parameters['b2'] = cp.zeros(output_size)

        self.layers = {}
        self.layers['Affine1'] = Affine(self.parameters['W1'], self.parameters['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.parameters['W2'], self.parameters['b2'])

        self.last_layer = Softmax_with_CE()

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
        self.loss(x, t)  # Some functions need forward before backward

        dLdx = 1
        dLdx = self.last_layer.backward(dLdx)
        for layer in reversed(self.layers.values()):
            dLdx = layer.backward(dLdx)
        
        gradients = {}
        gradients['W1'] = self.layers['Affine1'].dLdW
        gradients['b1'] = self.layers['Affine1'].dLdb
        gradients['W2'] = self.layers['Affine2'].dLdW
        gradients['b2'] = self.layers['Affine2'].dLdb
        return gradients


# Load MNIST dataset
with open("MNIST_flattened_onehot.pickle", "rb") as fr:
    x_train = pickle.load(fr)
    x_test = pickle.load(fr)
    t_train = pickle.load(fr)
    t_test = pickle.load(fr)

# Apply model to MNIST dataset
model = TwoLayerNN(784, 100, 10)
batch_size = 100
learning_rate = 0.1
loss_list = []

for _ in tqdm(range(1000)):
    batch_mask = cp.random.choice(len(x_train), batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    gradients = model.back_propagation(x_batch, t_batch)
    for key in model.parameters.keys():
        model.parameters[key] -= learning_rate * gradients[key]
    
    loss_list.append(model.loss(x_batch, t_batch))

# plot loss vs iteration
plt.figure()
plt.plot(loss_list)
plt.xlabel('iteration')
plt.ylabel('train error')
plt.show()