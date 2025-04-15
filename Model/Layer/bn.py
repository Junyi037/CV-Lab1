from .layer import Layer
import cupy as cp


class BN(Layer):
    """
        Batch Normalization Layer

        用于卷积层或全连接层的通用批归一化层。

        Args:
            normalized_dims (tuple): 需输入需要计算统计量的维度元组
            epsilon (float): 为防止分母为零，加入的小常数，默认为 1e-5，足够小

        示例：
            >>> layer = BN(normalized_dims=(0, 2, 3))  # 对除通道外的所有维度进行归一化
            >>> features = cp.random.randn(100, 3, 32, 32)  # (N, C, H, W)
            >>> (layer.forward(features)).shape
            (100, 3, 32, 32)
    """
    def __init__(self, normalized_dims, epsilon=1e-5):
        """
        需输入需要计算统计量的维度元组，例如：
        - 对于(N,C)输入使用(0,)
        - 对于(N,C,H,W)输入使用(0,2,3)
        """
        super().__init__(False)
        self.normalized_dims = normalized_dims  # 需计算统计量的维度
        self.epsilon = epsilon  # 小量，提高数值计算稳定性


    def forward(self, X, train=True):  # forward process 对于训练和预测相同
        dims = self.normalized_dims
        epsilon = self.epsilon

        # 计算每个特征的样本数
        M = cp.prod(cp.array([X.shape[d] for d in dims]))

        # 计算均值和方差
        mu = cp.mean(X, axis=dims, keepdims=True)
        var = cp.var(X, axis=dims, keepdims=True)

        # 归一化
        X_centered = X - mu
        std_inv = 1.0 / cp.sqrt(var + epsilon)
        X_norm = X_centered * std_inv

        self.cache = X_norm, std_inv, M
        return X_norm


    def backward(self, dY):
        dims = self.normalized_dims
        X_norm, std_inv, M = self.cache

        # 计算梯度
        dY_sum = cp.sum(dY, axis=dims, keepdims=True)
        dY_Xnorm_sum = cp.sum(dY * X_norm, axis=dims, keepdims=True)
        dX = (dY - dY_sum / M - X_norm * dY_Xnorm_sum / M) * std_inv

        # 清除缓存
        self.cache = None
        return dX