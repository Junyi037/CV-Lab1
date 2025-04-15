from .layer import Layer
import cupy as cp
from cupy.lib.stride_tricks import sliding_window_view as sliding


class Conv(Layer):
    """
        Same Convolution Layer with Stride = 1.

        步长为 1 的等宽卷积，用于提取特征。

        Args:
            C_in (int): 输入数据通道数
            C_out (int): 输出数据通道数
            K (int): 卷积核尺寸大小
            seed (int): 随机数种子，默认为 42，宇宙的答案

        示例：
            >>> layer = Conv(C_in=3, C_out=64, K=3)  # 等宽卷积
            >>> features = cp.random.randn(100, 3, 32, 32)  # (N, C, H, W)
            >>> (layer.forward(features)).shape
            (100, 3, 32, 32)
    """
    def __init__(self, C_in, C_out, K, seed=42):
        """
        W: (C_out, C_in, K, K)
        b: (C_out)
        """
        super().__init__(True)  # 可以更新
        self.C_in = C_in
        self.C_out = C_out
        self.K = K

        # He 初始化可更新参数
        cp.random.seed(seed)
        std = cp.sqrt(2.0 / C_in)
        self.params = {
            'W': cp.random.normal(0, std, (C_out, C_in, K, K)),
            'b': cp.zeros(C_out).reshape(1, -1, 1, 1)
        }


    def zero_pad(self, X):
        """
        等宽卷积，填充 (K-1)/2
        """
        if (self.K - 1) % 2 != 0:
            raise ValueError("卷积核尺寸 K 须为奇数。")
        padding = (self.K - 1) // 2

        X_pad = cp.pad(X, (
            (0, 0),  # 样本量
            (0, 0),  # 通道数
            (padding, padding),  # 高度
            (padding, padding)  # 宽度
        ), 'constant', constant_values=0)

        return X_pad

    def forward(self, X, train=True):  # forward process 对于训练和预测相同
        """
        X: (N, C_in, H_in, W_in)
        Y: (N, C_out, H_out, W_out)
        """
        W = self.params['W']
        b = self.params['b']  # 确保偏置 b 是 CuPy 数组
        K = self.K
        self.cache = X

        # 填充
        X = self.zero_pad(X)

        # 生成滑动窗口
        windows = sliding(X, (K, K), axis=(2, 3))  # (N, C_in, H_out, W_out, K, K)
        windows = windows.transpose(0, 2, 3, 1, 4, 5)  # (N, H_out, W_out, C_in, K, K)

        # Y = X * W + b
        Y = cp.einsum('nhwcij,ocij->nhwo', windows, W)  # (N, H_out, W_out, C_out)
        Y = Y.transpose(0, 3, 1, 2)  # (N, C_out, H_out, W_out)

        # 处理偏置 b
        Y = Y + b  # 此处无需 reshape，因为 b 已经是正确的形状 (1, C_out, 1, 1)

        return Y

    def backward(self, dY):
        """
        dY/grads_in: (N, C_out, H_out, W_out)
        dX: (N, C_in, H_in, W_in)
        """
        X = self.cache
        W = self.params['W']
        K = self.K


        # ------------------------------------- db ------------------------------------
        db = cp.sum(dY, axis=(0, 2, 3), keepdims=False).reshape(1, -1, 1, 1)
        self.grads['b'] = db


        # ------------------------------------- dW ------------------------------------
        ## 填充
        X_pad = self.zero_pad(X)

        ## 生成滑动窗口
        windows = sliding(X_pad, (K, K), axis=(2, 3))  # (N, C_in, H_out, W_out, K, K)
        windows = windows.transpose(0, 2, 3, 1, 4, 5)  # (N, H_out, W_out, C_in, K, K)

        ## dW = X_pad * dY  (convolution)
        grads_in_ = dY.transpose(0, 2, 3, 1)  # (N, H_out, W_out, C_out)
        dW = cp.einsum('nhwcij,nhwo->ocij', windows, grads_in_)  # (C_out, C_in, K, K)
        self.grads['W'] = dW


        # ------------------------------------- dX ------------------------------------
        ## 旋转卷积核
        W_rot = cp.rot90(W, k=2, axes=(2, 3))

        ## 填充接收梯度
        dY_pad = self.zero_pad(dY)

        ## 生成滑动窗口
        windows = sliding(dY_pad, (K, K), axis=(2, 3))  # (N, C_out, H_in, W_in, K, K)
        windows = windows.transpose(0, 2, 3, 1, 4, 5)  # (N, H_in, W_in, C_out, K, K)

        ## dX = dY_pad * rot(W)  (convolution)
        W_rot = W_rot.transpose(1, 0, 2, 3)  # (C_in, C_out, K, K)
        dX = cp.einsum('nhwoij,coij->nhwc', windows, W_rot)  # (N, H_in, W_in, C_in)
        dX = dX.transpose(0, 3, 1, 2)  # (N, C_in, H_in, W_in)

        # 清除缓存
        self.cache = None

        # 返回输入梯度 dX
        return dX
