import torch
import json

# Load json file
json_file = open(r'data.json', 'r')
config = json.load(json_file)

# Batch Size
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-4
# Let's check the device for cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Save model
def save_checkpoint(model, optimizer):
    print('==>!!!Saving Checkpoint!!!<==')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    location = r'D:\Deep_Learning\Algorithm\Audio_Processing\Project\Model\Model_Version_0.0.1.pth.tar'
    torch.save(checkpoint, location)


# Data paths
ANNOTATION_AUDIO = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\clips'
TRAIN_DATA_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv'
VALIDATION_DATA_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\validation.tsv'