import numpy as np


class Variable:
    """
    >>> data = np.array(1.0)
    >>> x = Variable(data)
    >>> print(x.data)
    1.0
    """

    def __init__(self, data) -> None:
        self.data = data
