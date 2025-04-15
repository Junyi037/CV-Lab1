import argparse
from torchvision.datasets import CIFAR10
from Model import augmentation, process, Model, compute_accuracy_and_loss

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Test a pre-trained model on CIFAR-10 dataset")
    parser.add_argument('--path', type=str, default='best_model_params.npz',
                        help="Path to the saved model parameters (default: 'experiments/best_model/best_model_params.npz')")
    return parser.parse_args()

# 获取命令行参数
args = parse_args()

# 加载数据
test_data = CIFAR10(root="./data", train=False, download=True)

# 数据增强和处理
test_aug = augmentation(flag='test')
test_data = process(test_data, test_aug)
test_images, test_labels_onehot, test_labels = test_data

# 实例化模型
model = Model(None)

# ------------------------------------------------- 面向用户部分 ----------------------------------------------------------
# 加载模型
model = model.load(args.path)  # 使用命令行输入的路径，若没有则使用默认路径
# ----------------------------------------------------------------------------------------------------------------------

# 测试模型
acc, loss = compute_accuracy_and_loss(model, test_images, test_labels, test_labels_onehot, 512)
print(f"Accuracy: {acc}, Loss: {loss}")

# 可视化所有层参数
model.visualize_parameters()
