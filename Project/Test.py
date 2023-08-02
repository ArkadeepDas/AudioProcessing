import torch
from Dataset import AudioTextData
from Audio_to_Text_Model import Audio_To_Text_Model
from torch.utils.data import DataLoader
import util

# Data paths
ANNOTATION_AUDIO = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\clips'
ANNOTATION_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\test.tsv'

# Let's load the data
audiotextdata = AudioTextData(Audio_Path=ANNOTATION_AUDIO,
                              Label_Path=ANNOTATION_PATH,
                              config=util.config)

audio, label = audiotextdata[0]
print('Audio File: ', audio)
print('Label: ', label)

# Let's load the model and optimizer
model = Audio_To_Text_Model()
optimizer = torch.optim.Adam(model.parameters(), lr=util.LEARNING_RATE)

# Let's load the checkpoint
load_checkpoint = torch.load(
    r'D:\Deep_Learning\Algorithm\Audio_Processing\Project\Model\Model_Version_0.0.1.pth.tar'
)
# Load weight and optimizer
model.load_state_dict(load_checkpoint['state_dict'])
optimizer.load_state_dict(load_checkpoint['optimizer'])
print('---!Load Data and Model Successfully!---')

# Convert model to test mode
model.eval()

data = DataLoader(audiotextdata, shuffle=True)
data_iter = iter(data)

# Fetch single data
audio, label = next(data_iter)
label = label.tolist()
output = model(audio)
# Take first output from batch
output = output[0]
print('Output Shape', output.shape)

# Calculate the maximum value
indices = torch.argmax(output, dim=-1)
print(indices.shape)
indices = indices.tolist()

# Predict the output text
text = ""
g = ''
characters = []
for i in indices:
    if i != 0:
        character = util.config['Number_to_Character'][str(i)]
        characters.append(character)
print(text.join(characters))