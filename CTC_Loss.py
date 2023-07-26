# Let's understand the CTC Loss
import torch
import pandas as pd
import torch.nn as nn
from torch.nn import functional

# Let's read a text data and capture the text
data = pd.read_csv(
    r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv',
    sep='\t')

sentences = data['sentence'].values.tolist()

# Let's chapture unique characters from our data
characters = []
for sentence in sentences:
    for char in sentence:
        characters.append(char)

characters = set(characters)
print(characters)
print('Total Characters: ', len(characters))

# Let's create the vocabulary
characters_to_number = dict()
for idx, character in enumerate(characters):
    characters_to_number[character] = idx + 1

print('Character to Number Map:', characters_to_number)

item_to_find = [1, 2, 3]
for keys, values in characters_to_number.items():
    if values in item_to_find:
        print('Require Keys: ', keys)


# Let's create the model
class CTCLossModel(nn.Module):

    def __init__(self, number_of_character):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=(3, 3),
                               padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(3, 3),
                               padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.lstm = nn.LSTM(input_size=16384, hidden_size=256, num_layers=2)
        self.output = nn.Linear(in_features=16384,
                                out_features=number_of_character)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[2])
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.output(x)
        return x


ctc_loss_model = CTCLossModel(115)
data = torch.randn((1, 3, 512, 512))
output = ctc_loss_model(data)

# CTC Loss
ctc_loss = nn.CTCLoss()
outputs = 'Log probability from the model for each time step (Batch size x Number of classes)'
targets = 'Ground truth label: (Maximum number of Character, Batch size)'
input_lenghths = 'We can think of it in different way, it if our input (Batchsize, Embedding, Features), then it will calculate that much embedding size. Suppose input is (4, 5, 10) -> input_lengths = [5, 5, 5, 5]'
target_lengths = 'Length of each target data. The characters of two sentences are: ([1, 2, 3, 4, 5],[2, 3]) : target_lengths -> [5,2]'
loss = ctc_loss(outputs, targets, input_lenghths, target_lengths)