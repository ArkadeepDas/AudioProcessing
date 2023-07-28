# Here we are going to create model for Audio to text transformation
# Model Structure -> Block1: Convolution -> Batch Normalization -> Relu
#                            Convolution -> Batch Normalization -> Relu
#                            Max Pooling
# 3 blocks of Block1

import torch
import torch.nn as nn


# Let's create the convolution block
class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cnn_1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)
        self.cnn_2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels * 2,
                               kernel_size=3,
                               padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels * 2)
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
class LSTMBlock(nn.Module):

    def __init__(self, input_size, number_of_classes=115):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=128,
                            num_layers=4)
        self.linear_1 = nn.Linear(in_features=32768, out_features=256)
        self.linear_2 = nn.Linear(in_features=256, out_features=512)
        self.linear_3 = nn.Linear(in_features=512, out_features=256)
        self.linear_4 = nn.Linear(in_features=256,
                                  out_features=number_of_classes)
        self.batch_norm_1 = nn.BatchNorm1d(256)
        self.batch_norm_2 = nn.BatchNorm1d(512)
        self.batch_norm_3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x


# Let's get together and create the model
class Audio_To_Text_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnnblock_1 = CNNBlock(in_channels=1, out_channels=8)
        self.cnnblock_2 = CNNBlock(in_channels=16, out_channels=32)
        self.cnnblock_3 = CNNBlock(in_channels=64, out_channels=128)
        self.lstmblock = LSTMBlock(input_size=4072, number_of_classes=115)

    def forward(self, x):
        x = self.cnnblock_1(x)
        x = self.cnnblock_2(x)
        x = self.cnnblock_3(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = self.lstmblock(x)
        return x


# data = torch.randn((1, 1, 64, 4073))
# model = Audio_To_Text_Model()
# model.eval()
# output = model(data)
# print(output.shape)