import numpy as np


class Variable:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):  # New
        self.creator = func

    def backward(self):
        f = self.creator
        if self.creator is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # New
        self.input = input
        self.output = output  # New
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, grad):  # grad from prev step
        x = self.input.data
        grad = (2 * x) * grad  # grad of current step
        return grad


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, grad):
        x = self.input.data
        grad = np.exp(x) * grad
        return grad


if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    # backprop
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)