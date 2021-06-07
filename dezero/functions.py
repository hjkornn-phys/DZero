import numpy as np
from dezero.core import Function, Variable, as_array, as_variable
from dezero import utils, cuda


class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, grad):
        (x,) = self.inputs
        grad = cos(x) * grad
        return grad


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, grad):
        (x,) = self.inputs
        grad = -sin(x) * grad
        return grad


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
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
        xp = cuda.get_array_module(x)
        y = xp.transpose(x)
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
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
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
        xp = cuda.get_array_module(x)
        y = 1 / (1 + xp.exp(-x))
        return y

    def backward(self, grad):
        y = self.outputs[0]()
        grad = grad * y * (1 - y)
        return grad


def sigmoid(x):
    return Sigmoid()(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        (x,) = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)
        if xp == np:
            xp.add.at(gx, self.slices, gy)  # gx + gy at index
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


# ================================== done by myself
class Softmax(Function):
    def __init__(self, axis=1) -> None:
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        if xp == np:
            x = x - xp.max(x, self.axis, keepdims=True)
        else:
            x = x - xp.amax(x, self.axis, keepdims=True)
        y = xp.exp(x)
        partition = xp.sum(y, self.axis, keepdims=True)
        y /= partition
        return y

    def backward(self, grad):
        y = self.outputs[0]()
        grad = y * grad
        dipart = grad.sum(axis=self.axis, keepdims=True)
        grad -= y * dipart
        return grad


def softmax(x, axis=1):
    return Softmax(axis)(x)


class CrossEntropyLoss(Function):
    def forward(self, y_pred, y_true):
        N = y_pred.shape[0]
        xp = cuda.get_array_module(y_pred)
        y_pred = xp.where(y_pred < 1e-15, 1e-15, y_pred)
        cce = -xp.sum(y_true * xp.log(y_pred)) / xp.float(N)
        return cce

    def backward(self, grad):
        y_pred, y_true = self.inputs
        N, D = y_pred.shape
        xp = cuda.get_array_module(y_pred)
        y_true_onehot = xp.eye(D, dtype=y_true.dtype)[y_true.data]
        grad *= (y_pred - y_true_onehot) / N
        return grad


def cross_entropy(y_pred, y_true):  # apply one-hot beforehand
    return CrossEntropyLoss()(y_pred, y_true)


def accuracy(pred, label):
    pred, label = as_variable(pred), as_variable(label)

    pred = pred.data.argmax(axis=1).reshape(label.shape)
    result = pred == label.data
    acc = result.mean()
    return Variable(as_array(acc))