# Let's understand the CTC Loss
import torch
import pandas as pd

# Let's read a text data and capture the text
data = pd.read_csv(
    r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv',
    sep='\t')

sentences = data['sentence']
print(sentences)