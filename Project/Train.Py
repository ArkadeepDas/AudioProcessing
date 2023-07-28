# In this file we are going to train our model

import json
import torch
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

# # Let's train the model
# for epoch in range(EPOCHS):
#     loop = tqdm(train_loader, leave=True)
#     for idx, (x, y) in enumerate(loop):
#         x, y = x.to(device), y.to(device)
