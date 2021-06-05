if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP
import dezero.datasets
import dezero.dataloader
import dezero.optimizers
from dezero import no_grad

MAX_EPOCH = 5000
BATCH_SIZE = 30
HIDDEN_SIZE = 10
lr = 1.0

train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_loader = dezero.dataloader.DataLoader(train_set, batch_size=BATCH_SIZE)
test_loader = dezero.dataloader.DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False
)

model = MLP((HIDDEN_SIZE, 3))
optimizer = dezero.optimizers.MomentumSGD().setup(model)

for epoch in range(MAX_EPOCH):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:  # Using Iterator
        y = F.softmax(model(x))
        t_onehot = np.eye(3)[t]
        t_onehot = Variable(t_onehot)
        loss = F.cross_entropy(y, t_onehot)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)  # mean cross entropy is returned by loss
        sum_acc += float(acc.data) * len(t)  # Number of correct prediction

    if not (epoch + 1) % 100:
        print(f"epoch:{epoch+1}")
        print(
            f"train loss: {sum_loss/len(train_set):.4f}, accuracy: {sum_acc/len(train_set):.4f} "
        )

    sum_loss, sum_acc = 0, 0
    with no_grad():
        for x, t in test_loader:  # Using Iterator
            y = F.softmax(model(x))
            t_onehot = np.eye(3)[t]
            t_onehot = Variable(t_onehot)
            loss = F.cross_entropy(y, t_onehot)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(
                t
            )  # mean cross entropy is returned by loss
            sum_acc += float(acc.data) * len(t)  # Number of correct prediction
        if not (epoch + 1) % 100:
            print(
                f"test loss: {sum_loss/len(test_set):.4f}, accuracy: {sum_acc/len(test_set):.4f} "
            )
