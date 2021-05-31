if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Function
from dezero import Variable
from dezero.utils import plot_dot_graph
import math


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, grad):
        x = self.inputs[0].data
        grad = np.cos(x) * grad
        return grad


def sin(x):
    return Sin()(x)


x = Variable(np.pi / 4)
y = sin(x)
y.backward()

print(y.data)
print(x.grad)


def my_sin(x, threshold=1e-140):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


x = Variable(np.pi / 4)
y = my_sin(x)
y.backward()

print(y.data)
print(x.grad)

x.name = "x"
y.name = "y"
plot_dot_graph(y, verbose=False, to_file="sin140.png")