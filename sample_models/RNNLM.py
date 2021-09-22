import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DLwithcp import functions, layers, optimizers


class RNNLM:
    def __init__(self, words_num, embd_vector_size, hidden_size):
        V, D, H = words_num, embd_vector_size, hidden_size

        self.parameters = {}
        self.parameters['Embd_W'] = cp.random.randn(V, D)
        self.parameters['RNN_W_h'] = cp.random.randn(H, H)
        self.parameters['RNN_W_x'] = cp.random.randn(D, H)
        self.parameters['RNN_b'] = cp.zeros(H)
        self.parameters['Aff_W'] = cp.random.randn(H, V) / cp.sqrt(H)
        self.parameters['Aff_b'] = cp.zeros(V)

        self.layers = {}
        self.layers['Embd'] = layers.Time_Embedding(self.parameters['Embd_W'])
        self.layers['RNN'] = layers.Time_RNN(self.parameters['RNN_W_h'], \
            self.parameters['RNN_W_x'], self.parameters['RNN_b'])
        self.layers['Affine'] = layers.Time_Affine(self.parameters['Aff_W'], \
            self.parameters['Aff_b'])
        
        self.last_layer = layers.Time_SoftmaxWithCrossEntropy()
    
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
with open("datasets\PTB_sequence.pickle", "rb") as fr:
    seq_train = pickle.load(fr)
    seq_valid = pickle.load(fr)
    seq_test = pickle.load(fr)
    id_to_word = pickle.load(fr)
    word_to_id = pickle.load(fr)

seq_train = seq_train[:1000]
x_train = seq_train[:-1]
t_train = seq_train[1:]

words_num = int(max(seq_train)) + 1
embd_vector_size = 100
hidden_size = 100

model = RNNLM(words_num, embd_vector_size, hidden_size)
optimizer = optimizers.SGD(learning_rate=0.1)
batch_size = 10
bptt_size = 5
max_iter = len(x_train) // (batch_size * bptt_size)
perplexity_list = []
epoch_list = []


for epoch in tqdm(range(100)):
    total_loss = 0
    for iter_ in range(max_iter):

        mask = list(range(iter_ * batch_size * bptt_size, \
            (iter_+1) * batch_size * bptt_size))
        x_batch = x_train[mask].reshape(batch_size, bptt_size)
        t_batch = t_train[mask].reshape(batch_size, bptt_size)
        gradients = model.back_propagation(x_batch, t_batch)
        optimizer.update(model.parameters, gradients)
        total_loss += model.loss_

    perplexity = cp.exp(total_loss / max_iter)
    perplexity_list.append(float(perplexity))
    epoch_list.append(epoch+1)


# Plot test perplexity vs epoch
plt.figure()
plt.plot(epoch_list, perplexity_list)
plt.xlabel('epoch')
plt.ylabel('perplexity')
plt.show()
