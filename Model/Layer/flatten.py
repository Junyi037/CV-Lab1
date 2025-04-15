from .layer import Layer


class Flatten(Layer):
    """
        Flatten Layer

        展平多维数据，置与卷积层输出之后，全连接输入之前。

        示例：
            >>> layer = Flatten()
            >>> features = cp.random.randn(100, 3, 32, 32)  # (N, C, H, W)
            >>> (layer.forward(features)).shape
            (100, 3072)
    """
    def __init__(self):
        super().__init__(False)


    def forward(self, X, train=True):
        """
        X: (N, C_in, H_in, W_in)
        X_flat: (N, C_in * H_in * W_in)
        """
        self.cache = X
        return X.reshape(X.shape[0], -1)


    def backward(self, grads_in):
        """
        dX_flat: (N, C_in * H_in * W_in)
        dX: (N, C_in, H_in, W_in)
        """
        return grads_in.reshape(self.cache.shape)
