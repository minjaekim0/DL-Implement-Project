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


class ReLU: # y = ReLU(x)
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


class Convolution: # O = I * F + B
    def __init__(self, F, B):
        self.F = F
        self.B = B
    
    def forward(self, I):
        self.I = I

        F = self.F
        B = self.B

        N, C, H, W = I.shape
        FN, _, FH, FW = F.shape
        OH, OW = H - FH + 1, W - FW + 1
        O = cp.zeros((N, FN, OH, OW))
        
        for n in range(N):
            for fn in range(FN):
                for oh in range(OH):
                    for ow in range(OW):
                        
                        for c in range(C):
                            for h in range(oh, oh+FH):
                                for w in range(ow, ow+FW):
                                    O[n, fn, oh, ow] += I[n, c, h, w] * F[fn, c, h-oh, w-ow]
                        O[n, fn, oh, ow] += B[fn]
        return O

    def backward(self, dLdO):
        I = self.I
        F = self.F
        B = self.B

        N, C, H, W = I.shape
        FN, _, FH, FW = F.shape
        OH, OW = H - FH + 1, W - FW + 1

        dLdB = cp.zeros(B.shape)
        for fn in range(FN):
            
            for n in range(N):
                for oh in range(OH):
                    for ow in range(OW):
                        dLdB[fn] += dLdO[n, fn, oh, ow]
        self.dLdB = dLdB

        dLdF = cp.zeros(F.shape)
        for fn in range(FN):
            for c in range(C):
                for fh in range(FH):
                    for fw in range(FW):
                        
                        for n in range(N):
                            for oh in range(OH):
                                for ow in range(OW):
                                    dLdF[fn, c, fh, fw] += dLdO[n, fn, oh, ow] * I[n, c, fh+oh, fw+ow]
        self.dLdF = dLdF

        dLdI = cp.zeros(I.shape)
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):

                        for fn in range(FN):
                            oh_min = max(h-FH+1, 0)
                            oh_max = min(h, OH-1)
                            ow_min = max(w-FW+1, 0)
                            ow_max = min(w, OW-1)
                            
                            for oh in range(oh_min, oh_max+1):
                                for ow in range(ow_min, ow_max+1):
                                    dLdI[n, c, h, w] += dLdO[n, fn, oh, ow] * F[fn, c, h-oh, w-ow]
        return dLdI


class Flatten: # y = Flatten(x)
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dLdy):
        return dLdy.reshape(self.x_shape)






class CNN:
    def __init__(self, input_dimension=(1, 28, 28), filters=7, filter_size=(5, 5), hidden_size=100, output_size=10):
        C, H, W = input_dimension
        FN = filters
        FH, FW = filter_size
        OH, OW = H - FH + 1, W - FW + 1
        
        self.parameters = {}
        self.parameters['W1'] = cp.random.randn(filters, input_dimension[0], filter_size[0], filter_size[1])  # filters
        self.parameters['b1'] = cp.zeros(filters)  # bias in convolution
        self.parameters['W2'] = cp.random.randn(FN*OH*OW, hidden_size)
        self.parameters['b2'] = cp.zeros(hidden_size)
        self.parameters['W3'] = cp.random.randn(hidden_size, output_size)
        self.parameters['b3'] = cp.zeros(output_size)

        self.layers = {}
        self.layers['Conv1'] = Convolution(self.parameters['W1'], self.parameters['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Flatten'] = Flatten()
        self.layers['Affine1'] = Affine(self.parameters['W2'], self.parameters['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine2'] = Affine(self.parameters['W3'], self.parameters['b3'])

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
with open("MNIST_onehot.pickle", "rb") as fr:
    x_train = pickle.load(fr)
    x_test = pickle.load(fr)
    t_train = pickle.load(fr)
    t_test = pickle.load(fr)

x_train = x_train.reshape(60000, 1, 28, 28)
x_test = x_test.reshape(10000, 1, 28, 28)

# Apply model to MNIST dataset
model = CNN()
batch_size = 3
learning_rate = 0.1
loss_list = []

for _ in tqdm(range(50)):
    batch_mask = cp.random.choice(len(x_train), batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    gradients = model.back_propagation(x_batch, t_batch)
    for key in model.parameters.keys():
        model.parameters[key] -= learning_rate * gradients[key]
    loss_list.append(model.loss_)

# plot loss vs iteration
plt.figure()
plt.plot(loss_list)
plt.xlabel('iteration')
plt.ylabel('train error')
plt.show()