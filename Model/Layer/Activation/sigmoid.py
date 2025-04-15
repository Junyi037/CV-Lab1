from ..layer import Layer
import cupy as cp

class Sigmoid(Layer):
    """
        Sigmoid Layer

        用于处理二分类问题。

        示例：
            >>> layer = Sigmoid()
            >>> features = cp.random.randn(100, 1)  # (N, 1)
            >>> (layer.forward(features)).shape
            (N, 1)
    """
    def __init__(self):
        super().__init__(False)  # 不能更新

    def forward(self, X, train=True):
        self.cache = X
        return 1 / (1 + cp.exp(-X))

    def backward(self, dY):
        sigmoid = 1 / (1 + cp.exp(-self.cache))  # 前向传递结果
        self.cache = None
        return sigmoid * (1 - sigmoid) * dY  # Sigmoid 的导数
