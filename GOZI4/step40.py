if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import *

""" x = np.array([[1, 2, 3], [4, 5, 6]])
y = sum_to(x, (1, 3))
print(y)

y = sum_to(x, (2, 1))
print(y) """

x_0 = Variable(np.array([1, 2, 3]))
x_1 = Variable(np.array([10]))
y = x_0 + x_1
print(y)

y.backward()
print(x_1.grad)