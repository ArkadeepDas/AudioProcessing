# Here we check max len of characters in a label data and add padding

import pandas as pd

# Label data path
ANNOTATION_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv'

# Load text data
annotations = pd.read_csv(ANNOTATION_PATH, sep='\t')

# Label data
label_data = annotations['sentence']
# Let's see the length of the data
print('Total Data: ', len(label_data))

# Let's see the charecter length of each text
max_length = 0
for sentence in label_data:
    if len(sentence) > max_length:
        max_length = len(sentence)
# Let's check the maximum length of the sentence in our dataset
print('Maximum length: ', max_length)

# So we will use RNN/GRU/LSTM with output layer of 115
# Let's add padding without changing the character to numeric values
updated_sentence = []
for sentence in label_data:
    sentence = sentence.ljust(max_length, 'x')
    updated_sentence.append(sentence)

for sentence in updated_sentence:
    assert max_length == len(sentence), "Lenght of each sentence is not equal"