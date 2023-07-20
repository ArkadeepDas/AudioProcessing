# Here we are working with Mel Spectrogram.
# It is a common feature extraction technique used in audio processing
# Step 1): Audio signal load and preprocess -> resampling the audio to consistant sample rate and normalize the data.
# Step 2): The audio is devided into small overlapping frames.
# Step 3): For each frame short-time fourier transformer is computed.
# etc...

# Output of the Mel Spectrogram is a 2D representation of the audio

import torchaudio
from Custome_Audio_Processing import AudioData
import torch

# Data set paths
ANNOTATION_AUDIO = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\clips'
ANNOTATION_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv'

# Create the object of the AudioData class
audio_data = AudioData(ANNOTATION_AUDIO, ANNOTATION_PATH)
# Get the first data
signal, sample_rate, label, metadata = audio_data[0]

# Number of samples(or data points) taken per second to represent and analog audio to it's digital form
SAMPLE_RATE = 16000

# Here we don't apply any normalization
# Audio may be mono may be stereo, they have two channels or more
# So after loading we mix it down to mono
# Sample rate of all the data must be equal
# For resampling the audio from one sample rate to another

# We apply only when original sample rate is different from target sample rate
resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
# Let's apply the resampling transformation
signal = resampler(signal)

# Let's mixdown to mono to mono
# What we want to do is aggregating multiple channels and mix them down to a single channel
# Using mean operation we aggregate them. Aggregrate them using number of channels
# signal -> (num_channels, samples) -> (2, 16000) Convert to -> (1, 16000)
signal = torch.mean(signal, dim=0, keepdim=True)

# Let's create a Mel Spectrogram transformation
# n_fft = number of points used for each sort-time furier transform
# hop_length = number of overlap between adjacent frames
# n_mels = number of Mel frequency bins used in the representation
# So basically n_mels convert the data from 1D to 2D representation
# num_frames = calculated using n_fft and hop_length
# Output shape = (n_mels, num_frames)
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                       n_fft=1024,
                                                       hop_length=512,
                                                       n_mels=64)

# Lets's apply the Mel Spectrogram
signal = mel_spectrogram(signal)
# Let's check the output and the shape
print('Audio Signal: ', signal)
print('Shape of audio signal: ', signal.shape)