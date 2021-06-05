if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import numpy as np
import dezero.functions as F
import dezero.layers as L


np.random.seed(0)
x = np.random.randn(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

model = Model()
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = model.l2(F.sigmoid(model.l1(x)))
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data  # type(p.grad) == Variable

    if not i % 1000:
        print(loss)