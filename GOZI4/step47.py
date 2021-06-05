if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
indices = np.array([0, 0, 1])
y = F.get_item(x, indices)
print(y)

y = x[1]
print(y)

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import numpy as np
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


model = MLP((10, 3))


x = Variable(np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]]))
t = np.array([2, 0, 1, 0])
y = F.softmax(model(x))
t = np.eye(3)[t]
t = Variable(t)
loss = F.cross_entropy(y, t)
print(y, t)

loss.backward(retain_grad=True)
print(loss.grad)
print(x.grad)