from .layer import Layer
import cupy as cp


class Dropout(Layer):
    """
        Dropout Layer

        放置于全连接之前，在训练时随机禁用部分神经元，从而防止过拟合。

        Args:
            rate (float): Dropout 层的丢弃率。每个神经元训练时被禁用的概率是 rate
            seed (int): 随机数种子，默认为 42，宇宙的答案

        示例：
            >>> layer = Dropout(rate=0.35)  # 放置于全连接层之前
            >>> features = cp.random.randn(100, 32)  # (N, features)
            >>> (layer.forward(features)).shape
            (100, 32)
    """
    def __init__(self, rate, seed=42):
        super().__init__(False)  # 不包含可优化参数
        self.rate = rate  # 丢失率
        self.seed = seed


    def forward(self, X, train=True):  # forward process 对于训练和预测不同
        if train:
            cp.random.seed(self.seed)
            # 生成掩码 mask
            mask = (cp.random.rand(*X.shape) > self.rate).astype(cp.float32)
            self.cache = mask
            return X * mask / (1.0 - self.rate)  # 保持期望输出不变
        else:
            # 在测试阶段，直接返回输入
            return X


    def backward(self, dY):
        mask = self.cache
        self.cache = None  # 清除缓存
        return dY * mask / (1.0 - self.rate)
