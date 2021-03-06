import numpy as np


class Variable:
    def __init__(self, data) -> None:
        """
        >>> x= Variable(0.5)
        TypeError: <class 'numpy.float64'> is not supported as an input.
        """
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported as an input.")

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # New

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []  # New
        seen = set()

        def add_func(f):
            if f not in seen:
                funcs.append(f)
                seen.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            grads_out = [output.grad for output in f.outputs]
            grads_in = f.backward(*grads_out)
            if not isinstance(grads_in, tuple):
                grads_in = (grads_in,)

            for x, grad_in in zip(f.inputs, grads_in):
                if x.grad is None:
                    x.grad = grad_in
                else:
                    x.grad = x.grad + grad_in

                if x.creator is not None:
                    add_func(x.creator)  # New

    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])  # New
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs  # must be tuple to have len()
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Add(Function):
    """
    >>> x0 = Variable(np.array(2))
    >>> x1 =  Variable(np.array(3))]
    >>> f = Add()
    >>> y =  f(x0, x1)
    >>> print(y.data)
    5
    """

    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, grad):
        return grad, grad


def add(x0, x1):
    """
    >>> x0 = Variable(np.array(200))
    >>> x1 = Variable(np.array(300))
    >>> y = add(x0, x1)
    >>> print(y.data)
    500
    """
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, grad):  # grad from prev step
        x = self.inputs[0].data
        grad = (2 * x) * grad  # grad of current step
        return grad


def square(x):
    f = Square()
    return f(x)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, grad):
        x = self.input.data
        grad = np.exp(x) * grad
        return grad


def exp(x):
    f = Exp()
    return f(x)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y.data)
    print(y.grad)
    print(x.grad)