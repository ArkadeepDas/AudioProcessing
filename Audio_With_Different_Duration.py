# When we are working with different deep learning algorithm we want to have training data that is fixed in shape.
# We have to fix the shape of all the data present there.

import torch
import torchaudio
from Custome_Audio_Processing import AudioData

NUM_SAMPLES = 22050

# Data set paths
ANNOTATION_AUDIO = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\clips'
ANNOTATION_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv'

# Create the object of the AudioData class
audio_data = AudioData(ANNOTATION_AUDIO, ANNOTATION_PATH)
# Get the first data
signal, sample_rate, label, metadata = audio_data[0]

# Applying previous operations
SAMPLE_RATE = 16000
resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
# Resampling the data
signal = resampler(signal)
# Normalize the data / convert to mono
signal = torch.mean(signal, dim=0, keepdim=True)

# Now the data shape of the first audio -> [64,221]

# Now we want every data to be in same shape
# Number of features must be same for every audio signal
# So we need to add padding in the right side. '0' padding
# If data is bigger then we are going to cut the data
# Tensor -> [number of channels, number of samples] -> [1, number of samples] here after converting to mono
# If [1, 50000] -> [1, 22050]
if signal.shape[1] > NUM_SAMPLES:
    signal = signal[:, :22050]

# If the length is less then we add 0 padding
if signal.shape[1] < NUM_SAMPLES:
    padded_data = NUM_SAMPLES - signal.shape[1]
    # (0, padded_data) -> add padding 'padded_data' 0's in the last
    # If (2, padded_data) -> add padding '2' 0's in the biginning
    signal = torch.nn.functional.pad(signal, (0, padded_data))

# Appliying Mel Spectrogram
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                       n_fft=1024,
                                                       hop_length=512,
                                                       n_mels=64)

# Lets's apply the Mel Spectrogram
signal = mel_spectrogram(signal)
# Let's check the output and the shape
print('Audio Signal: ', signal)
print('Shape of audio signal: ', signal.shape)