import os
import pickle
import cupy as cp
from tqdm import tqdm
from .Layer import Loss
from matplotlib import rcParams
from matplotlib import pyplot as plt
from .Data.dataloader import dataloader


class Model:
    """
        神经网络模型类。

        Args:
            layers (list): 用于组织神经网络层。
            optimizer (Optimizer): 存放模型的优化器。

        示例：
            >>> from Model.Opimizer import AdamW
            >>> from Model.Layer import Conv, Linear
            >>> layers = [Conv(C_in=3, C_out=32, K=3), Linear(C_in=32*32*32, C_out=10), Loss()]
            >>> optimizer = AdamW(lr=0.0005, weight_decay=0.04)
            >>> model= Model(layers=layers, optimizer=optimizer)
    """
    def __init__(self, layers=None, optimizer=None):
        self.layers = layers
        self.optimizer = optimizer

    def forward(self, inputs, train=True):
        X, y = inputs

        # 遍历每一层进行前向传播
        for layer in self.layers:
            # 损失层需要多接收一个参数
            if isinstance(layer, Loss):
                # 损失层返回 batch_y_pred, batch_mean_loss
                return layer.forward((X, y), train)
            else:
                X = layer.forward(X, train)

        raise TypeError("The last layer must be Loss.")

    def backward(self, grads):
        for layer in reversed(self.layers):
            grads = layer.backward(grads)

    def update(self):
        for layer in self.layers:
            # 判断是否包含可更新参数
            if layer.updatable:
                self.optimizer.step(layer.params, layer.grads)
                self.optimizer.zero_grad(layer.grads)

    def fit(self, features, labels, batch_size, epoch):
        """
        训练一个 epoch ，并输出每个 batch 的损失
        """
        # 计算批数
        num_batches = len(features) // batch_size

        with tqdm(total=num_batches, desc=f"Train Epoch {epoch+1}", unit=" batch") as pbar:
            for X, y in dataloader(features, labels, batch_size):
                # 将数据迁移到 GPU 上加速运算
                X, y = cp.array(X), cp.array(y)

                # 前向传播，损失层需要额外传入 y
                y_pred, loss = self.forward((X, y))

                # 反向传播，损失层不接收任何梯度
                self.backward(None)

                # 参数更新
                self.update()

                # 进度条更新
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)



    def predict(self, inputs, batch_size, train=True):
        """
        Predict function that handles input in smaller batches to avoid memory overflow.
        Accepts a batch_size argument to handle different batch sizes for prediction.
        """
        X, y = inputs
        # 迁移数据
        X, y = cp.array(X), cp.array(y)

        num_samples = X.shape[0]
        Y_pred = []
        Loss = []

        # 分批次进行预测
        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # 前向传播
            y_pred_batch, loss_batch = self.forward((X_batch, y_batch), train)
            Y_pred.append(y_pred_batch)
            Loss.append(loss_batch)

        # 拼接所有结果
        Y_pred = cp.concatenate(Y_pred, axis=0)
        return Y_pred, cp.mean(cp.asarray(Loss))


    def save(self, file_path):
        """
        保存整个模型（包括结构、参数和优化器状态）到指定路径
        :param file_path: 保存模型的文件路径
        """
        # 创建一个字典来保存整个模型的信息
        model_info = {
            'layers': self.layers,
            'optimizer': self.optimizer,
        }

        # 遍历所有层，检查并保存可更新的参数
        model_params = {}
        for idx, layer in enumerate(self.layers):
            if layer.updatable:
                converted_params = {}
                for k, v in layer.params.items():
                    # 更可靠的类型检查，确保将 CuPy 数组转换为 NumPy 数组
                    if isinstance(v, cp.ndarray):
                        converted_params[k] = v.get()  # 转换为 NumPy 数组
                    else:
                        converted_params[k] = v
                layer_name = f"{layer.__class__.__name__}_{idx}"
                model_params[layer_name] = converted_params

        # 将模型参数保存到字典中
        model_info['params'] = model_params

        # 保存整个模型信息到文件
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model_info, f)

        print(f"Model saved to: {file_path}")


    def load(self, file_path):
        """
        从文件加载整个模型（包括结构、参数和优化器状态）
        :param file_path: 存储模型的文件路径
        :return: 加载后的模型
        """
        # 从文件加载模型信息
        with open(file_path, 'rb') as f:
            model_info = pickle.load(f)

        # 恢复模型的层和优化器
        self.layers = model_info['layers']
        self.optimizer = model_info['optimizer']

        # 恢复每层的参数
        model_params = model_info['params']
        for layer_name, layer_params in model_params.items():
            # 获取层的索引，用于匹配每层
            layer_class_name, layer_idx = layer_name.rsplit('_', 1)
            layer_idx = int(layer_idx)

            # 获取层对象，确保索引与层名一致
            layer = self.layers[layer_idx]

            # 为该层赋值参数
            for param_name, param_value in layer_params.items():
                layer.params[param_name] = cp.array(param_value)  # 恢复为 CuPy 数组


        print(f"Model loaded from: {file_path}")
        return self

    def visualize_parameters(self, max_cols=4):
        """
        可视化所有可更新的网络参数，
        对于不同维度的参数采用不同的展示方式：
          - 1D 参数（如偏置）：直方图展示分布情况
          - 2D 参数（如全连接层权重）：热力图展示权重矩阵
          - 4D 参数（典型卷积层参数）：在一个窗口内同时展示所有卷积核和整体直方图，
            便于全面比较参数数值分布及卷积核细节

        参数:
            max_cols (int): 当展示4D参数时，每行最多显示的卷积核数量
        """
        # 配置中文字体，确保中文显示正常（系统需支持SimHei或Microsoft YaHei字体）
        rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        rcParams['axes.unicode_minus'] = False

        def plot_1d_parameter(data, title):
            """显示1D参数的直方图"""
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(data, bins=50, color='skyblue', edgecolor='black')
            ax.set_xlabel("参数值")
            ax.set_ylabel("频次")
            fig.suptitle(title, fontsize=12)
            plt.tight_layout()
            plt.show()

        def plot_2d_parameter(data, title):
            """使用热力图显示2D参数"""
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(data.T, cmap='viridis', aspect='auto')
            fig.colorbar(im, ax=ax, label="权重值")
            ax.set_xlabel("输入维度")
            ax.set_ylabel("输出维度")
            fig.suptitle(title, fontsize=12)
            plt.tight_layout()
            plt.show()

        def plot_4d_parameter(data, title, max_cols):
            """在同一窗口中同时展示整体直方图和每个卷积核的图像"""
            c_out, c_in = data.shape[:2]
            # 展平前两个维度，得到形状 (C_out * C_in, H, W)
            kernels = data.reshape(-1, *data.shape[2:])
            total_kernels = len(kernels)
            grid_rows = min((total_kernels + max_cols - 1) // max_cols, 8)  # 限制最多显示8行

            fig = plt.figure(figsize=(14, 6))
            fig.suptitle(title, fontsize=12)
            # 使用GridSpec布局：左侧显示直方图，右侧展示卷积核网格
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])

            # 左侧直方图：展示所有卷积核的数值分布
            ax_hist = fig.add_subplot(gs[0, 0])
            ax_hist.hist(kernels.flatten(), bins=50, color='gray', edgecolor='black')
            ax_hist.set_title("整体参数直方图", fontsize=10)
            ax_hist.set_xlabel("参数值", fontsize=8)
            ax_hist.set_ylabel("频次", fontsize=8)
            ax_hist.tick_params(labelsize=8)

            # 右侧卷积核网格：利用嵌套GridSpec构建子图
            gs_kernels = gs[0, 1].subgridspec(grid_rows, max_cols, wspace=0.3, hspace=0.3)
            for i, kernel in enumerate(kernels[:grid_rows * max_cols]):
                ax_kernel = fig.add_subplot(gs_kernels[i])
                ax_kernel.imshow(kernel, cmap='coolwarm')
                ax_kernel.axis('off')
                label = f"C:{i // c_in}/{i % c_in}"
                ax_kernel.set_title(label, fontsize=6)
            plt.tight_layout()
            plt.show()

        # 遍历每一层中可更新的参数
        for layer_idx, layer in enumerate(self.layers):
            if not layer.updatable:
                continue

            for param_name, param in layer.params.items():
                try:
                    data = param.get()  # 获取参数数据
                except Exception as e:
                    print(f"获取参数 {param_name} 时出错: {e}")
                    continue

                title = f"层 {layer_idx} ({layer.__class__.__name__})\n{param_name} {data.shape}"

                if data.ndim == 1:
                    plot_1d_parameter(data, title)
                elif data.ndim == 2:
                    plot_2d_parameter(data, title)
                elif data.ndim == 4:
                    plot_4d_parameter(data, title, max_cols)
                else:
                    print(f"跳过 {data.ndim}D 参数: {param_name}")



