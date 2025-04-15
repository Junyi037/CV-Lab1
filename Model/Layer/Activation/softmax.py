from ..layer import Layer
import cupy as cp

class Softmax(Layer):
    """
        Softmax Layer.

        用于处理多分类问题。与交叉熵结合后，作为 Loss 层位于 Model/Layer/loss.py 文件中。

        示例：
            >>> layer = Softmax()
            >>> features = cp.random.randn(100, 10)  # (N, Classes)
            >>> (layer.forward(features)).shape
            (N, Classes)
    """
    def __init__(self):
        super().__init__(False)  # 不能更新

    def forward(self, X, train=True):
        self.cache = X
        exp_values = cp.exp(X - cp.max(X, axis=1, keepdims=True))  # 防止溢出
        return exp_values / cp.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dY):
        # Softmax 的反向传播计算公式
        softmax = self.cache  # 先前计算的输出
        batch_size = softmax.shape[0]
        dX = cp.zeros_like(softmax)

        for i in range(batch_size):
            s = softmax[i, :]
            jacobian_matrix = cp.diagflat(s) - cp.outer(s, s)  # 雅可比矩阵
            dX[i, :] = cp.dot(jacobian_matrix, dY[i, :])  # 计算梯度

        self.cache = None
        return dX
