if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
import numpy as np
from dezero import Variable
from dezero import functions as F

x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
    logs.append(x.grad.data)
    grad_x = x.grad
    x.cleargrad()
    grad_x.backward(create_graph=True)

labels = ["y=sin(x)", "y'", "y''", "y'''"]

for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc="lower right")
plt.show()
