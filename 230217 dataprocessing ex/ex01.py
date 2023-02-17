import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=10,
                               kernel_size=5, stride=1)
        self.fc1 = nn.Linear(10*12*12, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        print("연산전 x.size >> ", x.size())
        # 연산전 x.size >>  torch.Size([10, 1, 20, 20])
        x = F.relu(self.conv1(x))
        print("conv1 연산 후 x.size >> ", x.size())
        # conv1 연산 후 x.size >>  torch.Size([10, 3, 16, 16])


cnn = CNN()
output = cnn(torch.randn(10, 1, 20, 20))
