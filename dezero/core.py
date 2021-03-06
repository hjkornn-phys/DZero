import numpy as np
import weakref
import contextlib
import dezero

try:
    import cupy

    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = np.ndarray


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


def as_array(x, array_module=np):  # cupy.ndarray + 1 works
    if np.isscalar(x):
        return array_module.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Variable:
    def __init__(self, data, name=None) -> None:
        if data is not None:
            if not isinstance(data, array_types):
                data = as_array(data)
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):
        return dezero.functions.transpose(self)

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen = set()

        def add_func(f):
            if f not in seen:
                funcs.append(f)
                seen.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            grads_out = [output().grad for output in f.outputs]

            with using_config("enable_backprop", create_graph):
                grads_in = f.backward(*grads_out)
                if not isinstance(grads_in, tuple):
                    grads_in = (grads_in,)

                for x, grad_in in zip(f.inputs, grads_in):
                    if x.grad is None:
                        x.grad = grad_in
                    else:
                        x.grad = x.grad + grad_in

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(input) for input in inputs]  # New
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]  # Error fix

    __array_priority__ = 200  # New

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Add(Function):
    """
    >>> x0 = Variable(np.array(2))
    >>> x1 =  Variable(np.array(3))
    >>> f = Add()
    >>> y =  f(x0, x1)
    >>> print(y.data)
    5
    """

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = (
            x0.shape,
            x1.shape,
        )  # shape saved for broadcasting(backprop)
        y = x0 + x1
        return y

    def backward(self, grad):
        grad_x0, grad_x1 = grad, grad
        if self.x0_shape != self.x1_shape:  # broadcasting occured
            grad_x0 = dezero.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = dezero.functions.sum_to(grad_x1, self.x1_shape)
        return grad_x0, grad_x1


def add(x0, x1):
    """
    >>> x0 = Variable(np.array(200))
    >>> x1 = Variable(np.array(300))
    >>> y = add(x0, x1)
    >>> print(y.data)
    500
    """
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = (
            x0.shape,
            x1.shape,
        )  # shape saved for broadcasting(backprop)
        y = x0 * x1
        return y

    def backward(self, grad):
        x0, x1 = self.inputs
        grad_x0 = grad * x1  # '*' is overriden by Mul -> computational graph is created
        grad_x1 = grad * x0  # when Function.__call__()
        if self.x0_shape != self.x1_shape:  # broadcasting occured
            grad_x0 = dezero.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = dezero.functions.sum_to(grad_x1, self.x1_shape)
        return grad_x0, grad_x1


def mul(x0, x1):
    """
    >>> x0 = Variable(np.array(200))
    >>> x1 = Variable(np.array(300))
    >>> y = mul(x0, x1)
    >>> print(y.data)
    60000
    >>> y.backward()
    >>> print(x0.grad)
    300
    """
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, grad):
        return -grad


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = (
            x0.shape,
            x1.shape,
        )  # shape saved for broadcasting(backprop)
        return x0 - x1

    def backward(self, grad):
        grad_x0, grad_x1 = grad, grad
        if self.x0_shape != self.x1_shape:  # broadcasting occured
            grad_x0 = dezero.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = dezero.functions.sum_to(grad_x1, self.x1_shape)
        return grad_x0, -grad_x1


def sub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))  # The former
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = (
            x0.shape,
            x1.shape,
        )  # shape saved for broadcasting(backprop)
        y = x0 / x1
        return y

    def backward(self, grad):
        x0, x1 = self.inputs
        grad_x0 = grad / x1
        grad_x1 = grad * (-x0 / x1 ** 2)
        if self.x0_shape != self.x1_shape:  # broadcasting occured
            grad_x0 = dezero.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = dezero.functions.sum_to(grad_x1, self.x1_shape)
        return grad_x0, grad_x1


def div(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c) -> None:
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, grad):
        x = self.inputs[0]
        c = self.c
        grad = grad * c * x ** (c - 1)
        return grad


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__mul__ = mul
    Variable.__add__ = add
    Variable.__rmul__ = mul
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = dezero.functions.get_item

    Variable.matmul = dezero.functions.matmul
    Variable.dot = dezero.functions.matmul
    # Variable.max = dezero.functions.max
    # Variable.min = dezero.functions.min


class Parameter(Variable):
    pass