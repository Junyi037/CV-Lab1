import cupy as cp
from .optimizer import Optimizer


class AdamW(Optimizer):
    """
        Adam Optimizer with Weight Decay Regularization

        带权重衰减的 Adam 优化器，本质上是 Adam 与 L2 正则化的结合。

        Args:
            lr (float): 初始学习率
            beta1 (float): 一阶矩估计的衰减率
            beta2 (float): 二阶矩估计的衰减率
            weight_decay (float): 权重衰减率，本质上是 L2 正则项系数
            eps (float): 提高数值稳定性的小量

        示例：
            >>> optimizer = AdamW(lr=0.0005, weight_decay=0.04)  # 其他参数为默认值
            >>> optimizer.step(params=layer.params, grads=layer.grads)
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=5e-4):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay  # 添加权重衰减参数
        self.t = 0  # 时间步
        self.m = {}
        self.v = {}


    def step(self, params: dict, grads: dict):
        self.t += 1
        lr = self.lr

        for key in params:
            # 为每一层的参数分配一个惟一的 ID 存储在优化器中
            param_id = id(params[key])

            if param_id not in self.m:
                self.m[param_id] = cp.zeros_like(params[key])
                self.v[param_id] = cp.zeros_like(params[key])

            # 计算动量与梯度平方的移动平均
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grads[key]
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grads[key] ** 2)

            # 偏差修正
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

            # 应用参数更新，包含权重衰减
            params[key] -= lr * m_hat / (cp.sqrt(v_hat) + self.eps) + lr * self.weight_decay * params[key]