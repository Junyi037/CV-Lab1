from .layer import Layer
import cupy as cp


class Linear(Layer):
    """
        Linear Layer

        全连接线性层，作为线性多分类器，结合 Softmax Layer 和 Cross Entropy 使用。

        Args:
            C_in (int): 输入数据的通道数
            C_out (int): 输出数据的通道数
            seed (int): 随机数种子，默认为 42，宇宙的答案

        示例：
            >>> layer = Linear(C_in=32, C_out=128)
            >>> features = cp.random.randn(100, 32)  # (N, C_in)
            >>> (layer.forward(features)).shape
            (100, 128)
    """
    def __init__(self, C_in, C_out, seed=42):
        """
        W: (C_in, C_out)
        b: (C_out)
        """
        super().__init__(True)  # 可以更新

        # He 初始化可更新参数，适用于 ReLU 激活函数
        cp.random.seed(seed)
        std = cp.sqrt(2.0 / C_in)
        self.params = {
            'W': cp.random.normal(0, std, (C_in, C_out)),
            'b': cp.zeros(C_out)
        }


    def forward(self, X, train=True):
        """
        X: (N, C_in)
        Y: (N, C_out)
        """
        self.cache = X
        # 批次前向传播
        Y = cp.dot(X, self.params['W']) + self.params['b']
        return Y


    def backward(self, dY):
        """
        dY: (N, C_out)
        dX: (N, C_in)
        """
        X = self.cache

        # 参数梯度
        self.grads = {
            'W': cp.dot(X.T, dY),
            'b': cp.sum(dY, axis=0)
        }

        # 输入梯度
        dX = cp.dot(dY, self.params['W'].T)

        # 释放缓存
        self.cache = None

        # 返回输入梯度 dX
        return dX
