import torch.nn as nn
import torch.nn.functional as F

class DemoNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=100, out_channels=1, kernel_size=3, padding=1)
        self.linear = nn.Linear(49, 10)

    def forward(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = F.max_pool2d(h, 2)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pool2d(h, 2)
        h = self.conv3(h)
        h = F.relu(h)
        h = h.view(-1, 49)
        h = self.linear(h)
        return h