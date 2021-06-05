if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import numpy as np
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


np.random.seed(0)
x = np.random.randn(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
model.plot(x, to_file="MLP.png")
optimizer = optimizers.SGD(lr)
optimizer = optimizer.setup(model)
# or optimizer = optimizers.SGD(lr).setup(model)
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if not i % 1000:
        print(loss)