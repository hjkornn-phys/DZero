if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable
from dezero.utils import get_dot_graph, plot_dot_graph


def f(x):
    y = x ** 4.0 - 2.0 * x ** 2.0
    return y


x = Variable(np.array(2.0))
y = f(x)
y.backward(retain_grad=True, create_graph=True)
plot_dot_graph(y, to_file="y.png")
print(x.grad)
x.name = "x"
y.name = "y"
grad_x = x.grad
grad_x.name = "gx"
x.cleargrad()
grad_x.backward()
print(x.grad)
