class Layer:
    """
        Basic Layer

        基类，用于统一神经网络层的形式，提高规范性。

        Args:
            updatable (bool): 用于判断是否包含可更新参数
    """
    def __init__(self, updatable):
        self.updatable = updatable  # 判断是否包含可更新参数
        self.cache = None  # 缓存
        self.params = {}
        self.grads = {}

    def forward(self, X, train=True):
        raise NotImplementedError

    def backward(self, dY):
        raise NotImplementedError
