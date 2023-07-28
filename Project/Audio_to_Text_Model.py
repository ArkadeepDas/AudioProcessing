# Here we are going to create model for Audio to text transformation
# Model Structure -> Block1: Convolution -> Batch Normalization -> Relu
#                            Convolution -> Batch Normalization -> Relu
#                            Max Pooling
# 3 blocks of Block1

import torch
import torch.nn as nn


# Let's create the convolution block
class CNNBlock(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.cnn_1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=32)
        self.cnn_2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.cnn_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        x = self.cnn_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        return x


# Let's create LSTM and Linear block
# class LSTMBlock(nn.Module):

#     def __init__(self):
#         super().__init__(number_0f_classes=115)
#         self.lstm = nn.LSTM(input_size = , hidden_size)
