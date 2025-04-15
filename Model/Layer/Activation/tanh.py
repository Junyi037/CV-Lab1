from ..layer import Layer
import cupy as cp

class Tanh(Layer):
    """
        Tanh Layer.

        不常用的激活函数。

        示例：
            >>> layer = Tanh()
            >>> features = cp.random.randn(100, 3, 32, 32)  # (N, C, H, W)
            >>> (layer.forward(features)).shape
            (N, C, H, W)
    """
    def __init__(self):
        super().__init__(False)  # 不能更新

    def forward(self, X, train=True):
        self.cache = X
        return cp.tanh(X)

    def backward(self, dY):
        tanh = cp.tanh(self.cache)  # 前向传递结果
        self.cache = None
        return (1 - tanh ** 2) * dY  # Tanh 的导数
