# from .step01 import Variable


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    """
    >>> x = Variable(np.array(10))
    >>> f = Square()
    >>> y = f(x)
    >>> print(type(y))
    <class '__main__.Variable'>
    >>> print(y.data)
    100
    """

    def forward(self, x):
        return x ** 2
