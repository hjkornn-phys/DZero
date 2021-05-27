import numpy as np


class Variable:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = None


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
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

    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)
    print(2 * np.exp(x.data ** 2) * np.exp(x.data ** 2) * (2 * x.data))
    print(2 * np.exp((2 * x.data)))

    assert x.grad == 2 * np.exp(x.data ** 2) * np.exp(x.data ** 2) * (2 * x.data)
