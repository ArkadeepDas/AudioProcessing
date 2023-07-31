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
EPOCHS = 100
LEARNING_RATE = 2e-4
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
# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Define the cost function
ctc_loss = nn.CTCLoss(blank=0)

# Let's train the model
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        # Pass data to model
        y_pred = model(x)
        y_pred = y_pred.permute(1, 0, 2).log_softmax(2).requires_grad_()
        # Let's calculate the parameters for CTCLoss
        input_lengths = torch.full(size=(BATCH_SIZE, ),
                                   fill_value=y_pred.size(0),
                                   dtype=torch.int32)
        target_lengths = torch.full(size=(BATCH_SIZE, ),
                                    fill_value=y.size(1),
                                    dtype=torch.int32)
        # Let's calculate the loss
        loss = ctc_loss(y_pred, y, input_lengths, target_lengths)
        print('\nLoss: ', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
