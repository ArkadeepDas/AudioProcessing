# Here we are goung to create class to handle our dataset
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json

# Let's read json file
json_file = open(r'data.json', 'r')
config = json.load(json_file)


class AudioTextData(Dataset):

    def __init__(self, Audio_Path, Label_Path, config):
        super().__init__()
        self.Audio_Path = Audio_Path
        self.annotations = pd.read_csv(Label_Path, sep='\t')
        self.config = config

    # Get the file name
    def _audio_sample_name(self, index):
        file_name = self.annotations['path'][index]
        return file_name

    # Sample audio data if require to the input audio
    def _sample_audio_data(self, audio_data, sample_rate):
        if sample_rate < config['MaximumSampleRate']:
            resampler = torchaudio.transforms.Resample(
                sample_rate, config['MaximumSampleRate'])
            audio_data = resampler(audio_data)
        return audio_data

    # Mix down to Mono channels
    def _mix_down_audio_data(self, audio_data):
        if audio_data.shape[0] > 1:
            audio_data = torch.mean(audio_data, dim=0, keepdim=True)
        return audio_data

    # Remove extra samples if require to the input audio
    def _cut_audio(self, audio_data):
        if audio_data.shape[1] > self.config['MaximumAudioSampleLength']:
            audio_data = audio_data[:, :self.
                                    config['MaximumAudioSampleLength']]
        return audio_data

    # Add padding to the input audio
    def _padding_rightside_audio(self, audio_data):
        if audio_data.shape[1] < self.config['MaximumAudioSampleLength']:
            num_missing_samples = self.config[
                'MaximumAudioSampleLength'] - audio_data.shape[0]
            last_dim_padding = (0, num_missing_samples)
            audio_data = torch.nn.functional.pad(audio_data, last_dim_padding)
        return audio_data

    # Apply Mel Spectrogram transformation to the input audio
    def _mel_spectrogram(self, audio_data):
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['MaximumSampleRate'],
            n_fft=1024,
            hop_length=512,
            n_mels=64)
        audio_data = mel_spectrogram(audio_data)
        return audio_data

    # Conver text to numeric values
    def _convert_text_to_numeric(self, text_data):
        character_to_number = self.config['Character_to_Number']
        text_to_numeric = []
        for chatacter in text_data:
            text_to_numeric.append(character_to_number[chatacter])
        return text_to_numeric

    # Add padding to the text
    def _padding_rightside_text(self, label_data):
        maximum_character_length = self.config['MaximumCharacterLength']
        if len(label_data) < maximum_character_length:
            num_missing_text = maximum_character_length - len(label_data)
            last_dim_padding = (0, num_missing_text)
            label_data = torch.nn.functional.pad(label_data, last_dim_padding)
        return label_data

    # Return total data length
    def __len__(self):
        return len(self.annotations)

    # Capture datas
    def __getitem__(self, index):
        # Get audio data
        audio_data = self.Audio_Path + '\\' + self._audio_sample_name(
            index=index)
        audio, sample_rate = torchaudio.load(audio_data)
        audio = self._sample_audio_data(audio_data=audio,
                                        sample_rate=sample_rate)
        audio = self._mix_down_audio_data(audio_data=audio)
        audio = self._cut_audio(audio_data=audio)
        audio = self._padding_rightside_audio(audio_data=audio)
        audio = self._mel_spectrogram(audio_data=audio)

        # Get text data
        text = self.annotations['sentence'][index]
        label = self._convert_text_to_numeric(text_data=text)
        label = torch.tensor(label)
        label = self._padding_rightside_text(label_data=label)

        return audio, label


# Let's test the class
ANNOTATION_AUDIO = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\clips'
ANNOTATION_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv'
audiotextdata = AudioTextData(Audio_Path=ANNOTATION_AUDIO,
                              Label_Path=ANNOTATION_PATH,
                              config=config)
audio, label = audiotextdata[0]
print('Audio file: ', audio)
print('Shape of the audio file: ', audio.shape)
print('Audio label', label)