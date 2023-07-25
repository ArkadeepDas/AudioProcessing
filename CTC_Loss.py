# Let's understand the CTC Loss
import torch
import pandas as pd
import torch.nn as nn

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
        self.flatten = nn.Flatten()
        # Check the shape from the previous layer and set the input features in linear layer
        self.linear = nn.Linear(in_features=1048576, out_features=64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2)
        self.output = nn.Linear(in_features=128,
                                out_features=number_of_character)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x, _ = self.lstm(x)
        x = self.output(x)
        print('Shape after LSTM layers: ', x.size())


ctc_loss_model = CTCLossModel(115)
data = torch.randn((1, 3, 512, 512))
ctc_loss_model(data)