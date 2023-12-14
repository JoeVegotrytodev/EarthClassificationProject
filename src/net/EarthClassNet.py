from torch import nn

from src.device.Using_device import get_device


class EarthClastNet(nn.Module):
    """Описываем нашу сеть"""

    def __init__(self):
        super(EarthClastNet, self).__init__()

        self.to(get_device())

        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, padding=5 // 2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=3 // 2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=3 // 2, stride=2)
        self.maxp = nn.MaxPool2d(4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxp(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxp(x)
        flattened = self.flatten(x)
        x = self.fc1(flattened)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
