import cupy as cp


class CosineAnnealingLR:
    """
        余弦退火调度策略

        Args:
            optimizer (Optimizer): 优化器
            T_max (int): 周期
            eta_min (float): 最低学习率
            last_epoch (int): 最新的 epoch

        示例：
            >>> scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
            >>> scheduler.step()
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.lr_max = optimizer.lr  # 设定初始学习率为 lr_max


    def step(self):
        self.last_epoch += 1
        lr = self.eta_min + 0.5 * (self.lr_max - self.eta_min) * (1 + cp.cos(cp.pi * self.last_epoch / self.T_max))
        self.optimizer.lr = lr
