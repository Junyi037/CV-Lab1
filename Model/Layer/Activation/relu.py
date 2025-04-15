from ..layer import Layer
import cupy as cp


class ReLU(Layer):
    """
        Rectified Linear Unit Layer

        常用的激活函数。

        示例：
            >>> layer = ReLU()
            >>> features = cp.random.randn(100, 3, 32, 32)  # (N, C, H, W)
            >>> (layer.forward(features)).shape
            (N, C, H, W)
    """
    def __init__(self):
        super().__init__(False)  # 不能更新


    def forward(self, X, train=True):
        self.cache = X
        return cp.maximum(0, X)


    def backward(self, dY):
        X = self.cache
        self.cache = None
        return (X > 0).astype(X.dtype) * dY
