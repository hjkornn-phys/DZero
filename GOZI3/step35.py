if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
import numpy as np
from dezero import Variable
from dezero import functions as F
from dezero.utils import plot_dot_graph

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = "x"
y.name = "y"
y.backward(create_graph=True)

iters = 4

for i in range(iters):
    grad_x = x.grad
    x.cleargrad()
    grad_x.backward(create_graph=True)

grad_x = x.grad
grad_x.name = "grad_x" + str(iters + 1)
plot_dot_graph(grad_x, to_file=f"tanh{iters+1}.png")
