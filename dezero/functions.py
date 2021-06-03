import numpy as np
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
    def __init__(self, axis=None, keepdims=False) -> None:
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


def sum(x, axis=None, keepdims=True):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, grad):
        grad = sum_to(grad, self.x_shape)
        return grad


def broadcast_to(x, shape):
    if shape == x.shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, grad):
        grad = broadcast_to(grad, self.x_shape)
        return grad


def sum_to(x, shape):
    if shape == x.shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, grad):
        x, W = self.inputs
        grad_x = matmul(grad, W.T)
        grad_W = matmul(x.T, grad)
        return grad_x, grad_W


def matmul(x, W):
    return MatMul()(x, W)


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, grad):
        x0, x1 = self.inputs
        diff = x0 - x1
        grad_x0 = grad * diff * (2.0 / len(diff))
        grad_x1 = -grad_x0
        return grad_x0, grad_x1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y


class Linear(Function):  # Why can't I delete t.data like simple ver.
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, grad):
        x, W, b = self.inputs
        grad_b = None if b.data is None else sum_to(grad, b.shape)
        grad_x = matmul(grad, W.T)
        grad_W = matmul(x.T, grad)
        return grad_x, grad_W, grad_b


def linear(x, W, b):
    return Linear()(x, W, b)


class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, grad):
        y = self.outputs[0]()
        grad = grad * y * (1 - y)
        return grad


def sigmoid(x):
    return Sigmoid()(x)