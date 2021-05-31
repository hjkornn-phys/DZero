if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable


x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)
grad_x = x.grad  # ay/ax at x = 2.0
x.cleargrad()


z = grad_x ** 3 + y
z.backward()
#  x.grad is az/ax at x = 2.0
print(x.grad)
