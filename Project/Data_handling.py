# Here we are going to create a json file to store constant values
# Map data from character to numeric and calculate maximum characters present in a sentence
import pandas as pd
import json

Train_Data = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv'
Test_Data = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\test.tsv'

train_sentences = pd.read_csv(Train_Data, sep='\t')
test_sentences = pd.read_csv(Test_Data, sep='\t')

# Concatinate both data
total_sentences = pd.concat(
    [train_sentences['sentence'], test_sentences['sentence']])

# Let's calculate the maximum length of the total dataset
max_length = 0
for sentence in total_sentences:
    if len(sentence) > max_length:
        max_length = len(sentence)

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

# Let's create the total dictionary
constant_data = dict()
constant_data['MaximumLength'] = max_length
constant_data['Character_to_Number'] = character_to_number
constant_data['Number_to_Character'] = number_to_character

# Let's create the json file
with open('data.json', 'w') as f:
    json.dump(constant_data, f)

print('...!Json Created!...')