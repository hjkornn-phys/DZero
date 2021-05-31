import numpy as np
from dezero.core import Function


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, grad):
        (x,) = self.inputs
        grad = cos(x) * grad
        return grad


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, grad):
        (x,) = self.inputs
        grad = -sin(x) * grad
        return grad


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, grad):
        y = self.outputs[0]()
        grad = grad * (1 - y * y)
        return grad


def tanh(x):
    return Tanh()(x)