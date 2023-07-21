# We take input label as text
# Let's convert the text to token so that it will became a numerical value

from transformers import BertTokenizer
from Custome_Audio_Processing import AudioData

# Data set paths
ANNOTATION_AUDIO = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\clips'
ANNOTATION_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv'

# Create the object of the AudioData class
audio_data = AudioData(ANNOTATION_AUDIO, ANNOTATION_PATH)
# Get the first data
signal, sample_rate, label, metadata = audio_data[0]

# Here we are working with text part. Label data is text data here. Let's understand the tokenization of text data.
text = label