# Let's understand the CTC Loss
import torch
import pandas as pd
from collections import Counter

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