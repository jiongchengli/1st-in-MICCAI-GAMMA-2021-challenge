from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, epochs, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.last_epoch = last_epoch
        self.power = power
        self.epochs = epochs
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.epochs) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]
