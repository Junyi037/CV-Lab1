from .layer import Layer


class SE(Layer):
    def __init__(self, input_channels, reduction_ratio=16):
        self.input_channels = input_channels
        self.reduction_ratio = reduction_ratio
        self.fc1 = cp.random.randn(input_channels, input_channels // reduction_ratio)
        self.fc2 = cp.random.randn(input_channels // reduction_ratio, input_channels)

    def forward(self, X):
        # Global average pooling
        avg_pool = cp.mean(X, axis=(2, 3), keepdims=True)  # (N, C, 1, 1)

        # Fully connected layers to compute channel attention
        x = cp.matmul(avg_pool, self.fc1)  # (N, C//reduction_ratio, 1, 1)
        x = cp.matmul(x, self.fc2)  # (N, C, 1, 1)

        # Sigmoid activation to get attention weights
        attention_weights = cp.sigmoid(x)  # (N, C, 1, 1)

        # Apply the attention weights to the input
        return X * attention_weights
