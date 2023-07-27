# Here we are going to create a json file to store constant values
# Map data from character to numeric, calculate maximum characters present in a sentence and maximum sample present in the audio data
import torch
import torchaudio
import pandas as pd
import json

Audio_Data_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\clips'
Train_Data = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv'
Test_Data = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\test.tsv'

SAMPLE_RATE = 32000

train_data = pd.read_csv(Train_Data, sep='\t')
test_data = pd.read_csv(Test_Data, sep='\t')

# Concatinate both data
total_sentences = pd.concat([train_data['sentence'], test_data['sentence']])

# Let's calculate the maximum length of the total dataset
max_character_length = 0
for sentence in total_sentences:
    if len(sentence) > max_character_length:
        max_character_length = len(sentence)

# Now let's create the set of unique character present in our total dataset
total_characters = []
for sentence in total_sentences:
    for character in sentence:
        total_characters.append(character)
unique_characters = set(total_characters)

# Let's create the dictionary for unique character
character_to_number = dict()
number_to_character = dict()
for idx, character in enumerate(unique_characters):
    character_to_number[character] = idx + 1
    number_to_character[idx + 1] = character

# Let's check the maximum number of samples present in our audio dataset
max_audio_sample = 0
train_audio_files = train_data['path']
test_audio_files = test_data['path']
for train_audio in train_audio_files:
    audio_file = Audio_Data_PATH + '\\' + train_audio
    # Load audio data
    audio, sample_rate = torchaudio.load(audio_file)
    # Resample if require
    if sample_rate < SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        audio = resampler(audio)
    # Convert to mono audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    # Calculate the maximum length audio sample
    if audio.shape[1] > max_audio_sample:
        max_audio_sample = audio.shape[1]


# Let's create the total dictionary
constant_data = dict()
constant_data['MaximumAudioSampleLength'] = max_audio_sample
constant_data['MaximumCharacterLength'] = max_character_length
constant_data['Character_to_Number'] = character_to_number
constant_data['Number_to_Character'] = number_to_character

# Let's create the json file
with open('data.json', 'w') as f:
    json.dump(constant_data, f)

print('...!Json Created!...')