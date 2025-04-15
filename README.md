本项目从零开始构建了 **双层卷积 + 全连接** 的三层神经网络，在 CIFAR-10 数据集上进行图像分类任务，测试集精度能够达到 $68.36\%$ 。这篇文档主要介绍了如何训练及测试模型，包含代码下载、模型下载、训练和测试。

## 1. Introduction

模型被封装到 `Model` 包中，包下含有 `Data`、`Layer`、`Optimizer`、`Strategy` 四个模块。`Data` 模块中包含自己实现的 DataLoader 和数据预处理部分；`Layer` 模块中包含涉及到了各个网络层和各种可选择激活函数；`Optimizer` 中包含 SGDW 和 AdamW；`strategy` 模块中包含所用到的早停策略和余弦退火调度策略。`Model` 包中还含有 `model.py` 类文件和 `pipeline.py` 流水线文件，前者定义了网络层的组织方式和模型的方法（训练、更新、预测、保存参数、读取参数等等），后者构建了一个自动训练模型、计算损失和精度、绘制曲线、根据实验时间自动创建文件夹并保存实验结果和模型参数的函数。

代码结构如下：
```
CV-Lab1/
├── experiments/(保存参数时自动创建文件夹)
├── Model/
│   ├── Data/
│   ├── Layer/
│   ├── Optimizer/
│   ├── Strategy/
│   ├── __init__.py
│   ├── model.py
│   ├── pipeline.py
├── .gitignore
├── README.md
├── test.py
├── train.py
└── best_model_params.npz(需自行下载)
```

## 2. Setup

首先需要将模型代码下载到本地：（$ 表示命令行提示符，复制时可忽略此符号）
```
$git clone https://github.com/Junyi037/CV-Lab1.git
```
再从 https://drive.google.com/file/d/1IYII9Z75gK-oJU_9BmKggKUgEfcNuW3-/view?usp=sharing 下载预训练模型参数，并将其放在与 `test.py` 文件同级的项目根目录下，位置如前所示。

## 3. Train

在项目文件夹路径下打开终端，输入如下指令，即可以默认配置进行训练，并进行参数可视化：
```
$python train.py
```
如果想要自定义超参数，可以先使用如下命令查看可选超参：

```
$python train.py --help
```

会显示如下可选择超参数：

```
options:
  -h, --help            show this help message and exit
  --lr LR               Initial learning rate for AdamW (default: 0.0002)
  --weight_decay WEIGHT_DECAY
                        Weight decay for AdamW (default: 0.004)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 512)
  --rate RATE           Dropout rate (default: 0.4)
  --epochs EPOCHS       Number of training epochs (default: 80)
  --T_max T_MAX         Cosine annealing T_max (default: 25)
  --eta_min ETA_MIN     Minimum learning rate for cosine annealing (default: 1e-05)
  --patience PATIENCE   Early stopping patience (default: 5)
  --delta DELTA         Minimum accuracy delta for early stopping (default: 0.0004)
  --layers              Display the docstrings of all relevant classes in Model.Layer
  --layer LAYER         Display the docstring of a specific class (e.g., Conv, Linear, etc.)
```

可以据此添加可选参数项，例如：

```
$python train.py --lr 0.0001 --batch_size 256 --epochs 100 --patience 10
```

如果想要更改模型架构，可以在源代码中找到面向用户部分，修改 `layers` 部分即可。可选用的网络层有：

```
Conv, Linear, BN, ReLU, Pooling, Flatten, Dropout, Loss, LeakyReLU, Sigmoid, Softmax, Tanh
```

使用如下命令查看所有网络层接收参数：

```
$python train.py --layers
```

如若只想查看某个网络层的用法，可以使用如下命令，其中 `Conv` 可以替换为任何想要查看用法的网络层：

```
$python train.py --layer Conv
```

仿照文件中的代码组织方式，自定义网络架构即可。相同的地方可以修改优化器、调度算法、学习率调度策略。可以在在 `Model/Strategy` 中查找可用策略，或自己实现。 最后按照第一步运行代码即可。训练好的模型模型会自动保存到 `experiments` 文件夹中，以训练开始的系统时间命名文件夹。

## 4. Test

在项目文件夹路径下打开终端，输入如下指令，即可以默认模型参数路径（`best_model_params.npz`）进行训练和参数可视化：

```
$python test.py
```

如果想要自定义模型参数路径，可以使用如下命令（ `--path` 后面的路径为可更改自定义路径）：

```
$python test.py --path 'experiments/experiment_time/model_params.npz'
```
