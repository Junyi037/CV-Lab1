import cupy as cp


class Optimizer:
    """
        Basic Opimizer.

        基类优化器。

        Args:
            lr (float): 学习率。
    """
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        raise NotImplementedError

    def zero_grad(self, grads):
        for key in grads:
            grads[key] = cp.zeros_like(grads[key])
        return grads
