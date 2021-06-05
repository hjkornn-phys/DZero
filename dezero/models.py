import dezero.layers as L
import dezero.functions as F
from dezero import utils


class Model(L.Layer):
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, to_file=to_file)


class MLP(Model):
    def __init__(self, fc_output_size, activation=F.sigmoid) -> None:
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_size):
            layer = L.Linear(out_size)
            setattr(
                self, "l" + str(i), layer
            )  # cf) Layer class, super().__setattr__(name, value)
            self.layers.append(layer)

    def forward(self, inputs):
        x = inputs
        for l in self.layers[:-1]:
            y = self.activation(l(x))
        return self.layers[-1](y)
