import cupy as cp


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, parameters, gradients):
        for key in parameters.keys():
            parameters[key] -= self.learning_rate * gradients[key]


class Adam:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, parameters, gradients):
        alpha = self.alpha
        beta1 = self.beta1
        beta2 = self.beta2
        m = self.m
        v = self.v
        t = self.t

        if m is None:
            m, v = {}, {}
            for key, value in parameters.items():
                m[key] = cp.zeros_like(value)
                v[key] = cp.zeros_like(value)
        
        self.t += 1
        
        for key in parameters.keys():
            m[key] = beta1 * m[key] + (1 - beta1) * gradients[key]
            v[key] = beta2 * v[key] + (1 - beta2) * (gradients[key] ** 2)
            
            parameters[key] -= alpha * ((1 - beta2 ** t) ** 1/2) / (1 - beta1 ** 2) \
                * m[key] / (cp.sqrt(v[key]) + 1e-8)
