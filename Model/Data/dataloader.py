import cupy as cp


def dataloader(features, labels, batch_size, shuffle=True, seed=42):
    """
        Customized DataLoader

        随即批次梯度下降算法所需要的 dataloader。

        Args:
            features (cp.array): 样本特征
            labels (cp.array): 样本标签
            batch_size (int): 批量大小
            shuffle (bool): 是否打乱。默认为 True，因为训练样本通常是按照类别排列的
            seed (int): 随机数种子，默认为 42，宇宙的答案

        Yield:
            features[index[i:j]], labels[index[i:j]]
            对返回使用 for 循环时，每次迭代返回一个批次 (batch_features, batch_labels)
    """
    features = cp.array(features)  # 使用 cupy 数组
    labels = cp.array(labels)      # 使用 cupy 数组

    n = len(features)
    index = cp.arange(n)

    cp.random.seed(seed)
    if shuffle:
        cp.random.shuffle(index)

    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        yield features[index[i:j]], labels[index[i:j]]

