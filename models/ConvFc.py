import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBn(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3,
                stride=1, padding=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                            stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net_10(nn.Module):
    def __init__(self):
        super(Net_10, self).__init__()
        self.conv1 = ConvBn(3, 16, stride=1)
        self.conv2 = ConvBn(16, 32, stride=1)
        self.conv3 = ConvBn(32, 64, stride=2)
        self.conv4 = ConvBn(64, 96, stride=2)
        self.conv5 = ConvBn(96, 128, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)  # 32-32
        x = self.conv2(x)  # 32-32
        x = self.conv3(x)  # 32-16
        x = self.conv4(x)  # 16-8
        x = self.conv5(x)  # 8-4
        # print(x.size())
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Net_5(nn.Module):
    def __init__(self):
        super(Net_5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x