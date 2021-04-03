import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy as cp
import pickle


class RNN: # t = h_prev * W_h + x * W_x + b, h = tanh(t)
    def __init__(self, W_h, W_x, b):
        self.W_h = W_h
        self.W_x = W_x
        self.b = b

    def forward(self, h_prev, x):
        self.h_prev = h_prev
        self.x = x

        t = cp.matmul(h_prev, self.W_h) + cp.matmul(x, self.W_x) + self.b
        self.h = cp.tanh(t)
        return self.h

    def backward(self, dLdh):
        dLdt = (1 - self.h ** 2) * dLdh

        self.dLdW_h = cp.matmul(self.h_prev.T, dLdt)
        self.dLdW_x = cp.matmul(self.x.T, dLdt)
        self.dLdb = cp.sum(dLdt, axis=0)

        dLdh_prev = cp.matmul(dLdt, self.W_h.T)
        dLdx = cp.matmul(dLdt, self.W_x.T)

        return dLdh_prev, dLdx


class Time_RNN:
    def __init__(self, W_h, W_x, b, stateful=False):
        self.W_h = W_h
        self.W_x = W_x
        self.b = b
        self.stateful=False
    
    def set_state(self, h_prev_batch):
        self.h_prev_batch = h_prev_batch
    
    def reset_state(self):
        self.h_prev_batch = None
    
    def forward(self, x_batch):
        N, T, D = x_batch.shape
        _, H = self.W_x.shape

        if not self.stateful or self.h_prev_batch == None:
            self.h_prev_batch = cp.zeros((N, H))

        self.layers = []
        h_batch = cp.empty((N, T, H))
        h = self.h_prev_batch

        for t in range(T):
            layer = RNN(self.W_h, self.W_x, self.b)
            h = layer.forward(h, x_batch[:, t, :])
            h_batch[:, t, :] = h
            self.layers.append(layer)
        
        return h_batch
    
    def backward(self, dLdh_batch):
        N, T, H = dLdh_batch.shape
        D, _ = self.W_x.shape

        dLdx_batch = cp.empty((N, T, D))
        dLdh = cp.zeros_like(self.h_prev_batch)

        self.dLdW_h = cp.zeros_like(self.W_h)
        self.dLdW_x = cp.zeros_like(self.W_x)
        self.dLdb = cp.zeros_like(self.b)

        for t in reversed(range(T)):
            layer = self.layers[t]
            dLdh, dLdx = layer.backward(dLdh_batch[:, t, :] + dLdh)
            dLdx_batch[:, t, :] = dLdx

            self.dLdW_h += layer.dLdW_h
            self.dLdW_x += layer.dLdW_x
            self.dLdb += layer.dLdb

        return dLdx_batch


class Time_Embedding: # O = Time_Embedding(I, W)
    def __init__(self, W):
        self.W = W
    
    def forward(self, I):
        self.I = I
        O = self.W[I]
        return O
    
    def backward(self, dLdO):
        N, T, D = dLdO.shape    

        self.dLdW = cp.zeros_like(self.W)
        for n in range(N):
            for t in range(T):
                self.dLdW[self.I[n, t]] += dLdO[n, t, :]


class Time_Affine: # A = xW + b
    def __init__(self, W, b):
        self.W = W
        self.b = b
    
    def forward(self, x):
        N, T, D = x.shape
        _, V = self.W.shape

        self.x_ = x.reshape(N*T, -1)
        A = cp.matmul(self.x_, self.W) + self.b
        
        return A.reshape(N, T, V)
    
    def backward(self, dLdA):
        N, T, V = dLdA.shape
        D, _ = self.W.shape

        dLdA_ = dLdA.reshape(N*T, -1)

        self.dLdW = cp.matmul(self.x_.T, dLdA_)
        self.dLdb = cp.sum(dLdA_, axis=0)
        dLdx_ = cp.matmul(dLdA_, self.W.T)
        return dLdx_.reshape(N, T, D)


def softmax(x):
    x = x - cp.max(x, axis=1).reshape(-1, 1)
    return cp.exp(x) / cp.sum(cp.exp(x), axis=1).reshape(-1, 1)

def CE(y, t):
    if y.ndim == 1:
        return -cp.sum(t * cp.log(y + 1e-5))
    else:
        return -cp.sum(t * cp.log(y + 1e-5)) / len(y)
    
class Time_Softmax_with_CE: # y = softmax(x), L = CE(y, t)
    def forward(self, x, t):
        self.x = x

        N, T, V = x.shape
        self.N = N
        self.T = T
        self.V = V

        x_ = x.reshape(N*T, V)
        self.y_ = softmax(x_)

        # t_: onehot vector set form of t
        self.t_ = cp.zeros_like(x_)
        for row in range(N):
            for col in range(T):
                self.t_[row * col, t[row, col]] = 1

        L = CE(self.y_, self.t_)
        return L
    
    def backward(self, dLdL=1):
        dLdx_ = (self.y_ - self.t_) / (self.N * self.T)
        dLdx = dLdx_.reshape(self.x.shape)
        return dLdx
        
    

class RNNLM:
    def __init__(self, words_num, embd_vector_size, hidden_size):
        V, D, H = words_num, embd_vector_size, hidden_size

        self.parameters = {}
        self.parameters['Embd_W'] = cp.random.randn(V, D)
        self.parameters['RNN_W_h'] = cp.random.randn(H, H)
        self.parameters['RNN_W_x'] = cp.random.randn(D, H)
        self.parameters['RNN_b'] = cp.zeros(H)
        self.parameters['Aff_W'] = cp.random.randn(H, V)
        self.parameters['Aff_b'] = cp.zeros(V)

        self.layers = {}
        self.layers['Embd'] = Time_Embedding(self.parameters['Embd_W'])
        self.layers['RNN'] = Time_RNN(self.parameters['RNN_W_h'], \
            self.parameters['RNN_W_x'], self.parameters['RNN_b'])
        self.layers['Affine'] = Time_Affine(self.parameters['Aff_W'], \
            self.parameters['Aff_b'])
        
        self.last_layer = Time_Softmax_with_CE()
    
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
        gradients['Embd_W'] = self.layers['Embd'].dLdW
        gradients['RNN_W_h'] = self.layers['RNN'].dLdW_h
        gradients['RNN_W_x'] = self.layers['RNN'].dLdW_x
        gradients['RNN_b'] = self.layers['RNN'].dLdb
        gradients['Aff_W'] = self.layers['Affine'].dLdW
        gradients['Aff_b'] = self.layers['Affine'].dLdb
        return gradients




# Load PTB dataset
with open("PTB_sequence.pickle", "rb") as fr:
    seq_train = pickle.load(fr)
    seq_valid = pickle.load(fr)
    seq_test = pickle.load(fr)
    id_to_word = pickle.load(fr)
    word_to_id = pickle.load(fr)
        
seq_train = seq_train[:1000]
words_num = int(max(seq_train)) + 1
embd_vector_size = 100
hidden_size = 100
model = RNNLM(words_num, embd_vector_size, hidden_size)

x_train = seq_train[:-1]
t_train = seq_train[1:]

batch_size = 10
bptt_size = 5
max_iter = len(x_train) // (batch_size * bptt_size)
learning_rate = 0.1
perplexity_list = []

for epoch in tqdm(range(100)):
    total_loss = 0
    for iter_ in range(max_iter):

        mask = list(range(iter_ * batch_size * bptt_size, \
            (iter_+1) * batch_size * bptt_size))
        x_batch = x_train[mask].reshape(batch_size, bptt_size)
        t_batch = t_train[mask].reshape(batch_size, bptt_size)

        gradients = model.back_propagation(x_batch, t_batch)

        # Stochastic Gradient Descent
        for key in model.parameters.keys():
            model.parameters[key] -= learning_rate * gradients[key]
        total_loss += model.loss_
    
    perplexity = cp.exp(total_loss / max_iter)
    perplexity_list.append(perplexity)


# Plot test perplexity vs iteration
plt.figure()
plt.plot(perplexity_list)
plt.xlabel('iteration')
plt.ylabel('perplexity')
plt.show()
