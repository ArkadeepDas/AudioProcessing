# Previously we see that we have audio with multiple channels
# Now after preprocessing the data and convert it to 3D of the data we apply CNN on it
# Dimentions: (Channels x Audio Data) -> (Channels x Features x Mel frequency bins)

import torch
from torch import nn


# Let's create CNN network
# 4 Convolution block -> Flatten Layer -> Linear layer -> Softmax
class CNNNetwork(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=128 * 5 * 4, out_features=10)
        self.softmax = nn.Softmax(dim=1)