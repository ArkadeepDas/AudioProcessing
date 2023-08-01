# In this file we are going to train our model

import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from Dataset import AudioTextData
from Audio_to_Text_Model import Audio_To_Text_Model
from tqdm import tqdm
import util

LOWEST_VALIDATION_LOSS = float('inf')
# Load data
train_data = AudioTextData(Audio_Path=util.ANNOTATION_AUDIO,
                           Label_Path=util.TRAIN_DATA_PATH,
                           config=util.config)
validation_data = AudioTextData(Audio_Path=util.ANNOTATION_AUDIO,
                                Label_Path=util.VALIDATION_DATA_PATH,
                                config=util.config)
train_loader = DataLoader(dataset=train_data,
                          batch_size=util.BATCH_SIZE,
                          shuffle=True)
validation_loader = DataLoader(dataset=validation_data,
                               batch_size=util.BATCH_SIZE,
                               shuffle=True)

# Load model
model = Audio_To_Text_Model()
model.to(device=util.device)
# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=util.LEARNING_RATE)
# Define the cost function
ctc_loss = nn.CTCLoss(blank=0)

# Let's train the model
for epoch in range(util.EPOCHS):
    loop = tqdm(train_loader, leave=True)
    # Set model to training mode
    model.train()
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(util.device), y.to(util.device)
        # Pass data to model
        y_pred = model(x)
        y_pred = y_pred.permute(1, 0, 2).log_softmax(2).requires_grad_()
        # Let's calculate the parameters for CTCLoss
        input_lengths = torch.full(size=(y_pred.size(1), ),
                                   fill_value=y_pred.size(0),
                                   dtype=torch.int32)
        target_lengths = torch.full(size=(y.size(0), ),
                                    fill_value=y.size(1),
                                    dtype=torch.int32)
        # Let's calculate the loss
        training_loss = ctc_loss(y_pred, y, input_lengths, target_lengths)
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

    # Set model to validation mode
    model.eval()
    validation_loss = 0.0

    with torch.no_grad():
        for audio, label in validation_loader:
            audio, label = audio.to(util.device), label.to(util.device)
            prediction = model(audio)
            prediction = prediction.permute(1, 0, 2).log_softmax(2)
            in_lengths = torch.full(size=(prediction.size(1), ),
                                    fill_value=prediction.size(0),
                                    dtype=torch.int32)
            tar_lengths = torch.full(size=(label.size(0), ),
                                     fill_value=label.size(1),
                                     dtype=torch.int32)
            # Calculate the loss
            loss = ctc_loss(prediction, label, in_lengths, tar_lengths)
            validation_loss += loss.item()

    # Average loss
    validation_loss /= len(validation_loader)
    # Let's check the losses
    print(
        f'Epoch : {epoch}/{util.EPOCHS} - Training Loss: {training_loss.item()} - Validation Loss: {validation_loss}'
    )

    if validation_loss < LOWEST_VALIDATION_LOSS:
        LOWEST_VALIDATION_LOSS = validation_loss
        util.save_checkpoint(model=model, optimizer=optimizer)
