import numpy as np
from numpy.lib.stride_tricks import broadcast_to
from dezero.core import Function, as_variable
from dezero import utils


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


class Reshape(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, grad):
        return reshape(grad, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)  # Assure Variable instance
    return Reshape(shape)(
        x
    )  # as_variable is called automatically, since Reshape inherits Function


class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y

    def backward(self, grad):
        grad = transpose(grad)
        return grad


def transpose(x):
    return Transpose()(x)


class Sum(Function):
    def __init__(self, axis, keepdims) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)  # np.sum
        return y

    def backward(self, grad):
        grad = utils.reshape_sum_backward(grad, self.x_shape, self.axis, self.keepdims)
        grad = broadcast_to(grad, self.x_shape)
        return grad


def sum(x, axis, keepdims):
    return Sum(axis, keepdims)(x)
