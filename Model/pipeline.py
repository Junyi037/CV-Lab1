import os
import time
import cupy as cp
import matplotlib.pyplot as plt


def compute_accuracy_and_loss(model, images, labels, labels_onehot, batch_size):
    """
        计算准确率和损失的通用函数。

        Args:
            model (Model): 模型。
            images (np.array): 图像。
            labels (list): 标签。
            labels_onehot (np.array): 标签独热编码。
            batch_size (int): 批次大小。
    """
    images, labels, labels_onehot = cp.array(images), cp.array(labels), cp.array(labels_onehot)

    # 假设 model.predict 返回的是 (y_pred, loss_pred)
    y_pred, loss_pred = model.predict((images, labels_onehot), batch_size=batch_size, train=False)
    idx = cp.argmax(y_pred, axis=1)  # 预测的标签
    labels_cp = cp.array(labels)  # 原始标签
    acc = cp.mean((idx == labels_cp).astype(cp.float32))  # 计算准确率
    return float(acc.get()), float(loss_pred)


def save_experiment_plots(train_acc_list, test_acc_list, train_loss_list, test_loss_list, epochs, hyperparams, experiment_dir):
    """绘制并保存准确率和损失图表"""
    # 转换CuPy数组为NumPy数组
    def convert_cupy_to_numpy(data_list):
        return [x.get() if isinstance(x, cp.ndarray) else x for x in data_list]

    # 应用转换
    train_acc_list = convert_cupy_to_numpy(train_acc_list)
    test_acc_list = convert_cupy_to_numpy(test_acc_list)
    train_loss_list = convert_cupy_to_numpy(train_loss_list)
    test_loss_list = convert_cupy_to_numpy(test_loss_list)

    epochs_range = range(1, epochs + 1)

    # 设置共同标题（包含超参数信息）
    hyperparam_str_for_title = ', '.join([f"{key}: {value}" for key, value in hyperparams.items()])

    # 绘制图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Training and Test Results\n{hyperparam_str_for_title}', fontsize=14)

    # 绘制准确率曲线
    axes[0].plot(epochs_range, train_acc_list, label='Train Accuracy')
    axes[0].plot(epochs_range, test_acc_list, label='Test Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs. Epoch')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制损失曲线
    axes[1].plot(epochs_range, train_loss_list, label='Train Loss')
    axes[1].plot(epochs_range, test_loss_list, label='Test Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss vs. Epoch')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，以避免标题被遮挡
    plt.savefig(os.path.join(experiment_dir, f"accuracy_plot.png"), bbox_inches='tight')
    plt.show()


def pipeline(model, train_data, test_data, epochs, batch_size, hyperparams, scheduler, strategy):
    """
        训练和测试模型，并保存实验结果。

        Args:
            model (Model): 模型。
            train_data (tuple): 训练数据。
            test_data (tuple): 测试数据。
            epochs (int): 训练轮数。
            batch_size (int): 批次大小。
            hyperparams (dict): 超参数，用于绘图。
            scheduler (CosineAnnealingLR): （余弦退火）调度算法。
            strategy (EarlyStopping): （早停）策略。

        使用示例：
            依次传入参数即可。
    """
    # 获取当前系统时间（精确到秒）
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # 创建实验存储目录
    experiment_dir = os.path.join('experiments', timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # 获取超参数信息并创建文件名
    hyperparam_str = '_'.join([f"{key}_{value}" for key, value in hyperparams.items()])
    file_suffix = f"lr_{hyperparam_str}_epochs_{epochs}_batch_{batch_size}"

    # 训练和测试数据准备
    train_images, train_labels_onehot, train_labels = train_data
    test_images, test_labels_onehot, test_labels = test_data

    # 用于记录每个 Epoch 的准确率和损失
    train_acc_list, train_loss_list = [], []
    test_acc_list, test_loss_list = [], []

    total_epochs = epochs

    for epoch in range(epochs):
        print(f"\n------- Training Epoch {epoch + 1} -------")
        # 使用余弦退火策略
        scheduler.step()

        # 拟合数据
        model.fit(features=train_images, labels=train_labels_onehot, batch_size=batch_size, epoch=epoch)

        # 训练集预测计算准确率
        train_acc, train_loss = compute_accuracy_and_loss(model, train_images, train_labels, train_labels_onehot, batch_size)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # 测试集预测计算准确率
        test_acc, test_loss = compute_accuracy_and_loss(model, test_images, test_labels, test_labels_onehot, batch_size)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        # 判断是否早停
        if strategy.step(test_acc):
            total_epochs = epoch + 1
            break

    print("\n\nTraining completed!")

    # -------------------------------- 绘图 -----------------------------------
    save_experiment_plots(train_acc_list, test_acc_list, train_loss_list, test_loss_list, total_epochs, hyperparams, experiment_dir)

    # ---------------------------- 保存模型参数 ---------------------------
    model.save(os.path.join(experiment_dir, f"model_params.npz"))

    print(f"Results and model parameters saved to: {experiment_dir}")
