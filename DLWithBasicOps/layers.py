import cupy as cp
from DLWithBasicOps import functions


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


class SoftmaxWithCrossEntropy: # y = softmax(x), L = cross_entropy(y, t)
    def forward(self, x, t):
        self.y = functions.softmax(x)
        self.t = t
        self.L = functions.cross_entropy(self.y, self.t)
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
        
        """
        # slow method

        for n in range(N):
            for fn in range(FN):
                for oh in range(OH):
                    for ow in range(OW):
                        
                        for c in range(C):
                            for h in range(oh, oh+FH):
                                for w in range(ow, ow+FW):
                                    O[n, fn, oh, ow] += I[n, c, h, w] * F[fn, c, h-oh, w-ow]
                        O[n, fn, oh, ow] += B[fn]
        """

        I_ = cp.zeros((N*OH*OW, C*FH*FW))
        F_ = cp.zeros((C*FH*FW, FN))
        B_ = cp.zeros((N*OH*OW, FN))

        for n in range(N):
            for oh in range(OH):
                for ow in range(OW):
                    I_[n*OH*OW + oh*OW + ow] = I[n, 0:C, oh:oh+FH, ow:ow+FW].reshape(-1)

        for fn in range(FN):
            F_[:, fn] = F[fn, 0:C, 0:FH, 0:FW].reshape(-1)
            B_[:, fn] = B[fn]

        O_ = cp.matmul(I_, F_) + B_

        for n in range(N):
            for fn in range(FN):
                O[n, fn] = O_[n*OH*OW : (n+1)*OH*OW, fn].reshape(OH, OW)

        return O

    def backward(self, dLdO):
        I = self.I
        F = self.F
        B = self.B

        N, C, H, W = I.shape
        FN, _, FH, FW = F.shape
        OH, OW = H - FH + 1, W - FW + 1

        dLdB = cp.zeros_like(B)
        dLdF = cp.zeros_like(F)
        dLdI = cp.zeros_like(I)

        """
        # slow methods

        for fn in range(FN):
            
            for n in range(N):
                for oh in range(OH):
                    for ow in range(OW):
                        dLdB[fn] += dLdO[n, fn, oh, ow]
        self.dLdB = dLdB
        
        for fn in range(FN):
            for c in range(C):
                for fh in range(FH):
                    for fw in range(FW):
                        
                        for n in range(N):
                            for oh in range(OH):
                                for ow in range(OW):
                                    dLdF[fn, c, fh, fw] += dLdO[n, fn, oh, ow] * I[n, c, fh+oh, fw+ow]
        self.dLdF = dLdF
        
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
        
        """

        # Calculate dLdB
        # Reshaped matrix -> Add _B in the end of the name
        dLdO_B = cp.zeros((FN, N*OH*OW))
        one_B = cp.ones((N*OH*OW, 1))

        for fn in range(FN):
            dLdO_B[fn] = dLdO[0:N, fn, 0:OH, 0:OW].reshape(-1)
        
        dLdB_B = cp.matmul(dLdO_B, one_B)
        dLdB = dLdB_B.reshape(-1)
        self.dLdB = dLdB


        # Calculate dLdF
        # Reshaped matrix -> Add _F in the end of the name
        dLdO_F = dLdO_B
        I_F = cp.zeros((N*OH*OW, C*FH*FW))
        
        for c in range(C):
            for fh in range(FH):
                for fw in range(FW):
                    I_F[:, c*FH*FW + fh*FW + fw] = \
                        I[0:N, c, fh:fh+OH, fw:fw+OW].reshape(-1)
        
        dLdF_F = cp.matmul(dLdO_F, I_F)

        for fn in range(FN):
            dLdF[fn] = dLdF_F[fn].reshape(C, FH, FW)
        
        self.dLdF = dLdF


        # Calculate dLdI
        # Reshaped matrix -> Add _I in the end of the name
        dLdO_padding = cp.zeros((N, FN, OH+2*FH-2, OW+2*FW-2))
        dLdO_padding[:, :, FH-1:FH+OH-1, FW-1:FW+OW-1] = dLdO

        dLdO_I = cp.zeros((N*H*W, FN*FH*FW))
        F_I = cp.zeros((FN*FH*FW, C))

        for n in range(N):
            for h in range(H):
                for w in range(W):
                    dLdO_I[n*H*W + h*W + w] = \
                        dLdO_padding[n, 0:FN, h:h+FH, w:w+FW].reshape(-1)
                    # h-FH+1:h+1 -> add FH-1 to both
                    # w-FW+1:w+1 -> add FW-1 to both
        
        for c in range(C):
            F_I[:, c] = F[0:FN, c, 0:FH, 0:FW].reshape(-1)[::-1]
        
        dLdI_I = cp.matmul(dLdO_I, F_I)

        for n in range(N):
            for c in range(C):
                dLdI[n, c] = dLdI_I[n*H*W : (n+1)*H*W, c].reshape(H, W)

        return dLdI


class Pooling: # Maximum Pooling, O = Pooling(I)
    def __init__(self, filter_size):
        self.filter_size = filter_size

    def forward(self, I):
        self.I = I
        N, C, H, W = I.shape
        FH, FW = self.filter_size
        X, Y = int(H/FH), int(W/FW)

        I_ = cp.zeros((N*C*X*Y, FH*FW))

        for n in range(N):
            for c in range(C):
                for x in range(X):
                    for y in range(Y):
                        row = n*C*X*Y + c*X*Y + x*Y + y
                        I_[row] = I[n, c, x*FH:(x+1)*FH, y*FW:(y+1)*FW].reshape(-1)
        
        self.I_mask = I_.argmax(axis=1)
        I_ = I_.max(axis=1)
        O = I_.reshape(N, C, X, Y)
        return O
    
    def backward(self, dLdO):
        I = self.I
        N, C, H, W = I.shape
        FH, FW = self.filter_size
        X, Y = int(H/FH), int(W/FW)

        dLdI_ = cp.zeros((N*C*X*Y, FH*FW))

        for n in range(N):
            for c in range(C):
                for x in range(X):
                    for y in range(Y):
                        row = n*C*X*Y + c*X*Y + x*Y + y
                        dLdI_[row, self.I_mask[row]] = dLdO[n, c, x, y]
        
        dLdI = dLdI_.reshape(N, C, H, W)
        return dLdI


class Flatten: # y = Flatten(x)
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dLdy):
        return dLdy.reshape(self.x_shape)


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


class Time_SoftmaxWithCrossEntropy: # y = softmax(x), L = cross_entropy(y, t)
    def forward(self, x, t):
        self.x = x

        N, T, V = x.shape
        self.N = N
        self.T = T
        self.V = V

        x_ = x.reshape(N*T, V)
        self.y_ = functions.softmax(x_)

        # t_: onehot vector set form of t
        self.t_ = cp.zeros_like(x_)
        for row in range(N):
            for col in range(T):
                self.t_[row * col, t[row, col]] = 1

        L = functions.cross_entropy(self.y_, self.t_)
        return L
    
    def backward(self, dLdL=1):
        dLdx_ = (self.y_ - self.t_) / (self.N * self.T)
        dLdx = dLdx_.reshape(self.x.shape)
        return dLdx
