import cupy as cp


def softmax(x):
    x = x - cp.max(x, axis=1).reshape(-1, 1)
    return cp.exp(x) / cp.sum(cp.exp(x), axis=1).reshape(-1, 1)

def cross_entropy(y, t):
    if y.ndim == 1:
        return -cp.sum(t * cp.log(y + 1e-5))
    else:
        return -cp.sum(t * cp.log(y + 1e-5)) / len(y)
