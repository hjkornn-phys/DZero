# from .step01 import Variable
# from .step02 import Function, Square
import numpy as np


class Exp(Function):
    """
    >>> A = Square()
    >>> B = Exp()
    >>> C = Square()

    >>> x = Variable(np.array(0.5))
    >>> a = A(x)
    >>> b = B(a)
    >>> y = C(b)
    >>> print(y.data)
    1.648721270700128
    """

    def forward(self, x):
        return np.exp(x)
