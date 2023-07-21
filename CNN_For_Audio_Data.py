# Previously we see that we have audio with multiple channels
# Now after preprocessing the data and convert it to 3D of the data we apply CNN on it
# Dimentions: (Channels x Audio Data) -> (Channels x Features/frequency axis x Mel frequency bins)

# Here we assume that it's a classification problem. The model will predict 10 classes.

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
        # Check the shape from the previous layer and set the input features in linear layer
        self.linear = nn.Linear(in_features=128 * 4 * 2, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Applying 4 Convolution layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Applying Flatten layer
        x = self.flatten(x)
        # Applying Linear layer
        x = self.linear(x)
        # Applying softmax
        output = self.softmax(x)
        return output


# Let's test the model
def test():
    # Input data
    x = torch.randn((1, 1, 64, 44))
    model = CNNNetwork()
    pred = model(x)
    print(pred.shape)


if __name__ == '__main__':
    test()