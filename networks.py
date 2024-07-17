import torch.nn as nn

class DemoNet(nn.Module):
    """
    A Simple Pytorch Module for demo
    """

    def __init__(self):
        super().__init__()
        self.test_1 = nn.Linear(784, 2048)
        self.te_2st = nn.Linear(2048, 784)
        self._3test = nn.Linear(784, 10)

    def forward(self, x):
        h = self.test_1(x)
        h = F.mish(h)
        h = self.te_2st(h)
        h = x + h
        h = self._3test(h)
        return h