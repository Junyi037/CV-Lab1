from .layer import Layer
from cupy.lib.stride_tricks import sliding_window_view as sliding


class Pooling(Layer):
    """
        Pooling Layer

        池化层，降低分辨率，减少线性层的参数量，从而减轻了模型训练负担。

        Args:
            K (int): 卷积核的尺寸。会使分辨率降低为原来的 1/K

        示例：
            >>> layer = Pooling()  # 默认将分辨率降低一半
            >>> features = cp.random.randn(100, 3, 32, 32)  # (N, C, H, W)
            >>> (layer.forward(features)).shape
            (100, 3, 16, 16)
    """
    def __init__(self, K=2):
        super().__init__(False)  # 不能更新
        self.K = K


    def forward(self, X, train=True):
        """
        X: (batch_size, C_in, H_in, W_in)
        """
        K = self.K
        H_in = X.shape[-2]
        if H_in % K != 0:
            raise ValueError("H_in 必须能被 K 整除。")

        # 生成滑动窗口
        windows = sliding(X, (K, K), axis=(2, 3))
        windows = windows[:, :, ::K, ::K, :, :]  # (N, C_in/C_out, H_out, W_out, K, K)

        X_pooling = windows.max(axis=(-2, -1), keepdims=True)
        mask = (windows == X_pooling)  # broadcasting

        self.cache = X.shape, mask
        return X_pooling.squeeze(-1).squeeze(-1)


    def backward(self, grads_in):
        """
        grads_in: (N, C_out, H_out, W_out)
        """
        shape, mask = self.cache

        grads_in_expanded = grads_in[..., None, None] * mask
        grads_in_expanded = grads_in_expanded.transpose(0, 1, 2, 4, 3, 5)
        grads_out = grads_in_expanded.reshape(shape)

        self.cache = None
        return grads_out
