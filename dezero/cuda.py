import numpy as np
from numpy.core.numeric import isscalar

gpu_enable = True
try:
    import cupy as cp

    cupy = cp
except ImportError:
    gpu_enable = False

from dezero import Variable


def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if isscalar(x):
        return np.array(x)

    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception("Cannot load CuPy")
    return cp.asarray(x)