import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def augmentation(flag):
    """
        CIFAR-10 数据增强

        在保持图像可识别性的前提下，通过温和增强提升小模型泛化能力。
        但事实上，由于网络较浅，较强的数据增强虽然会增强泛化能力，但会降低拟合能力，最终使精度下降。
        因此，仅保留了最简单的随机水平翻转。

        Args:
             flag (str): 'train' or 'test'

        Returns:
            transforms.Compose([...])
            根据 flag 值，针对训练集或是测试集返回不同的数据增强函数
    """

    if flag == 'train':
        return transforms.Compose([
            # # 填充4像素后随机裁剪（原图32x32 -> 填充至40x40 -> 裁剪回32x32）
            # transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            # 水平翻转（50%概率）
            transforms.RandomHorizontalFlip(p=0.5),
            # # 轻微颜色抖动（亮度/对比度/饱和度扰动控制在10%内）
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            # # 10%概率转灰度（保留大部分颜色信息）
            # transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            # 标准化（CIFAR-10统计量）
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.247, 0.243, 0.261]
            ),
            # # 轻量遮挡（Cutout，在标准化后的Tensor上操作）
            # transforms.Lambda(lambda x: cutout(x, n_holes=1, length=8)),
        ])
    elif flag == 'test':
        return transforms.Compose([
            transforms.ToTensor(),
        # 标准化（CIFAR-10统计量）
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.247, 0.243, 0.261]
        )
        ])
    else:
        raise ValueError('Flag must be train or test.')


# Cutout工具函数（需单独定义）
def cutout(tensor, n_holes=1, length=8):
    """
        在张量上随机遮挡正方形区域

        Args:
            tensor (Tensor): 输入张量 (C, H, W)
            n_holes (int): 遮挡区域数量
            length (int): 遮挡边长

        Returns:
            tensor * mask
            标志是否被遮挡的掩码
    """

    h, w = tensor.size(1), tensor.size(2)
    mask = torch.ones(h, w)

    for _ in range(n_holes):
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()

        y1 = max(0, y - length // 2)
        y2 = min(h, y + length // 2)
        x1 = max(0, x - length // 2)
        x2 = min(w, x + length // 2)

        mask[y1:y2, x1:x2] = 0

    return tensor * mask




def process(dataset, augmentation):
    """
        数据预处理

        Args:
            dataset (torchvision.datasets): 借助 torchvision 读取的数据集
            augmentation (func): 数据增强函数

        Returns:
            images, labels_onehot, labels
            图像，标签独热编码，标签
    """
    # 数据增强
    images = []
    for img in dataset.data:
        # 转换为 PIL 格式
        img = Image.fromarray(img)
        # 转换为 numpy 格式
        img = augmentation(img).numpy()  # (C, H, W)
        images.append(img)
    images = np.stack(images)  # (N, C, H, W)

    # 标签处理，转换为 One-hot 编码标签
    labels = dataset.targets

    labels_onehot = np.zeros((len(labels), 10))
    for n, _class in enumerate(labels):
        labels_onehot[n][_class] = 1

    return images, labels_onehot, labels
