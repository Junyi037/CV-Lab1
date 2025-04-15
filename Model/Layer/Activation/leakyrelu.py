from ..layer import Layer
import cupy as cp

class LeakyReLU(Layer):
    """
        Leaky Rectified Linear Unit Layer

        解决传统 ReLU 的神经元死亡问题。

        Args:
            alpha (float): 控制负值区域的斜率系数，默认为 0.01，不大也不小

        示例:
            >>> layer = LeakyReLU()
            >>> features = cp.random.randn(100, 3, 32, 32)  # (N, C, H, W)
            >>> (layer.forward(features)).shape
            (N, C, H, W)
    """
    def __init__(self, alpha=0.01):
        super().__init__(False)  # 不能更新
        self.alpha = alpha  # 设置负区域的斜率

    def forward(self, X, train=True):
        self.cache = X
        return cp.where(X > 0, X, self.alpha * X)

    def backward(self, dY):
        X = self.cache
        self.cache = None
        return (X > 0).astype(X.dtype) * dY + (X <= 0).astype(X.dtype) * self.alpha * dY  # LeakyReLU 的导数
