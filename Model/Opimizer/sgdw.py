from .optimizer import Optimizer


class SGDW(Optimizer):
    """
        SGD Opimizer with Weight Decay Regularization

        带权重衰减的 SGD 优化器，本质上是 SGD 与 L2 正则化的结合。

        Args:
            lr (float): 学习率
            weight_decay (float): 权重衰减率，本质上是 L2 正则项系数

        示例：
            >>> optimizer = SGDW(lr=0.0005, weight_decay=0.04)
            >>> optimizer.step(params=layer.params, grads=layer.grads)
    """
    def __init__(self, lr=0.01, weight_decay=0.05):
        super().__init__(lr)
        self.weight_decay = weight_decay


    def step(self, params, grads):
        lr = self.lr
        weight_decay = self.weight_decay

        for param in params:
            params[param] = (1 - lr * weight_decay) * params[param] - lr * grads[param]
        return params
