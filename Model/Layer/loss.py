from .layer import Layer
import cupy as cp


class Loss(Layer):
    """
        Loss Layer

        损失层。为了提高数值稳定性，将 Softmax 和 Cross Entropy 结合在一起，简化了反向传播的计算。

        示例：
            >>> layer = Loss()
            >>> features = cp.random.randn(100, 64)  # (N, features)
            >>> Y_pred, loss = layer.forward(features)
    """
    def __init__(self):
        super().__init__(False)  # 不能更新


    def forward(self, inputs, train=True):
        X, Y = inputs

        # 计算 softmax
        X_max = cp.max(X, axis=1, keepdims=True)
        X = X - X_max  # 提高数值稳定性
        exp_X = cp.exp(X)
        sum_exp = cp.sum(exp_X, axis=1, keepdims=True)
        Y_pred = exp_X / sum_exp

        self.cache = Y_pred, Y  # 缓存，用于反向传播

        # 计算损失
        log_probs = X - cp.log(sum_exp)
        loss = -cp.sum(Y * log_probs, axis=1)

        return Y_pred, cp.mean(loss)


    def backward(self, grads_in=None):
        Y_pred, Y = self.cache
        batch_size = Y_pred.shape[0]

        # 计算梯度
        dX = (Y_pred - Y) / batch_size
        return dX
