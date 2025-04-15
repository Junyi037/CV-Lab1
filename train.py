import argparse
from torchvision.datasets import CIFAR10
from Model import augmentation, process, Model
from Model.Layer import Conv, Linear, BN, ReLU, Pooling, Flatten, Dropout, Loss, LeakyReLU, Sigmoid, Softmax, Tanh
from Model.Opimizer import AdamW
from Model.Strategy import *
from Model import pipeline


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10 dataset with customizable hyperparameters")

    # 超参数部分
    parser.add_argument('--lr', type=float, default=0.0002, help="Initial learning rate for AdamW (default: 0.0002)")
    parser.add_argument('--weight_decay', type=float, default=0.004, help="Weight decay for AdamW (default: 0.004)")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for training (default: 512)")
    parser.add_argument('--rate', type=float, default=0.4, help="Dropout rate (default: 0.4)")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs (default: 80)")
    parser.add_argument('--T_max', type=int, default=25, help="Cosine annealing T_max (default: 25)")
    parser.add_argument('--eta_min', type=float, default=1e-05,
                        help="Minimum learning rate for cosine annealing (default: 1e-05)")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument('--delta', type=float, default=0.0004,
                        help="Minimum accuracy delta for early stopping (default: 0.0004)")

    # 添加查看文档字符串功能
    parser.add_argument('--layers', action='store_true',
                        help="Display the docstrings of all relevant classes in Model.Layer")

    # 查看特定类的文档字符串
    parser.add_argument('--layer', type=str,
                        help="Display the docstring of a specific class (e.g., Conv, Linear, etc.)")

    return parser.parse_args()


# 获取命令行参数
args = parse_args()

# 如果需要查看所有类的 __doc__ 注释
if args.layers:
    # 打印所有相关类的文档字符串
    classes = [Conv, Linear, BN, ReLU, Pooling, Flatten, Dropout, Loss, LeakyReLU, Sigmoid, Softmax, Tanh]
    for cls in classes:
        print(f"Class {cls.__name__} docstring:")
        print(cls.__doc__)
        print('-' * 80)  # 分隔符
    exit()

# 如果需要查看特定类的 __doc__ 注释
if args.layer:
    # 将传入的类名转换为对应的类对象
    classes_dict = {
        'Conv': Conv,
        'Linear': Linear,
        'BN': BN,
        'ReLU': ReLU,
        'Pooling': Pooling,
        'Flatten': Flatten,
        'Dropout': Dropout,
        'Loss': Loss,
        'LeakyReLU': LeakyReLU,
        'Sigmoid': Sigmoid,
        'Softmax': Softmax,
        'Tanh': Tanh
    }

    class_name = args.layer
    if class_name in classes_dict:
        cls = classes_dict[class_name]
        print(f"Class {cls.__name__} docstring:")
        print(cls.__doc__)
    else:
        print(f"Error: Class {class_name} not found.")
    exit()

# 加载数据
train_data = CIFAR10(root="./data", train=True, download=True)
test_data = CIFAR10(root="./data", train=False, download=True)

# 数据增强和处理
train_aug = augmentation(flag='train')
test_aug = augmentation(flag='test')
train_data = process(train_data, train_aug)
test_data = process(test_data, test_aug)

train_images, train_labels_onehot, train_labels = train_data
test_images, test_labels_onehot, test_labels = test_data
print(f"Train images shape: {train_images.shape}")
print(f"Train labels onehot shape: {train_labels_onehot.shape}")
print(f"Train labels shape: {len(train_labels)}")


# 使用命令行参数设置超参数
hyperparams = {
    'lr': args.lr,
    'weight_decay': args.weight_decay,
    'batch_size': args.batch_size,
    'rate': args.rate,
    'epochs': args.epochs,
    'T_max': args.T_max,
    'eta_min': args.eta_min,
    'patience': args.patience,
    'delta': args.delta
}




# ------------------------------------------------- 面向用户部分 ----------------------------------------------------------
# 模型架构
layers = [
    Conv(C_in=3, C_out=128, K=3),
    BN(normalized_dims=(0, 2, 3)),
    ReLU(),
    Pooling(K=2),

    Conv(C_in=128, C_out=256, K=3),
    BN(normalized_dims=(0, 2, 3)),
    ReLU(),
    Pooling(K=2),

    Flatten(),
    Dropout(rate=hyperparams['rate']),
    Linear(C_in=256 * 8 * 8, C_out=10),
    Loss()
]

# 定义优化器
optimizer = AdamW(lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])
# 定义调度算法
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=hyperparams['T_max'], eta_min=hyperparams['eta_min'])
# 定义早停策略
strategy = EarlyStopping(patience=hyperparams['patience'], delta=hyperparams['delta'])

# 建立模型
model = Model(layers=layers, optimizer=optimizer)
# ----------------------------------------------------------------------------------------------------------------------







# 训练模型
pipeline(model, train_data, test_data, hyperparams['epochs'], hyperparams['batch_size'], hyperparams,
         scheduler=scheduler, strategy=strategy)

# 可视化所有层参数
model.visualize_parameters()
