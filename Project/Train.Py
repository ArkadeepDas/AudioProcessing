# In this file we are going to train our model

import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from Dataset import AudioTextData
from Audio_to_Text_Model import Audio_To_Text_Model
from tqdm import tqdm

# Load json file
json_file = open(r'data.json', 'r')
config = json.load(json_file)

# Batch Size
BATCH_SIZE = 16
EPOCHS = 1

# Let's check the device for cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Data paths
ANNOTATION_AUDIO = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\clips'
TRAIN_DATA_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv'
VALIDATION_DATA_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\validation.tsv'
# Load data
train_data = AudioTextData(Audio_Path=ANNOTATION_AUDIO,
                           Label_Path=TRAIN_DATA_PATH,
                           config=config)
validation_data = AudioTextData(Audio_Path=ANNOTATION_AUDIO,
                                Label_Path=VALIDATION_DATA_PATH,
                                config=config)
train_loader = DataLoader(dataset=train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
validation_loader = DataLoader(dataset=validation_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

# Load model
model = Audio_To_Text_Model()
# Set model to training mode
model.train()
model.to(device=device)
# Define the cost function
ctc_loss = nn.CTCLoss()

# Let's train the model
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        # Pass data to model
        y_pred = model(x)
        print(y.shape)
        print(y_pred.shape)
        # Let's calculate the parameters for CTCLoss
        input_lengths = torch.full(size=(16, ),
                                   fill_value=115,
                                   dtype=torch.long)
        target_lengths = torch.randint(low=1,
                                       high=115,
                                       size=(16, ),
                                       dtype=torch.long)
        print(input_lengths.shape)
        print(target_lengths.shape)
        # Let's calculate the loss
        loss = ctc_loss(y_pred, y, input_lengths, target_lengths)
