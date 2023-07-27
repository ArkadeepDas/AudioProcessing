# Here we are goung to create class to handle our dataset
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd

SAMPLE_RATE = 32000


class AudioData(Dataset):

    def __init__(self, Audio_Path, Label_Path):
        super().__init__()
        self.Audio_Path = Audio_Path
        self.annotations = pd.read_csv(Label_Path, sep='\t')

    # Get the file name
    def _audio_sample_name(self, index):
        file_name = self.annotations['path'][index]
        return file_name

    # Sample audio data if require
    def _sample_audio_data(self, audio_data, sample_rate):
        if sample_rate < SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate,
                                                       SAMPLE_RATE)
            audio_data = resampler(audio_data)
        return audio_data

    # Return total data length
    def __len__(self):
        return len(self.annotations)

    # Capture datas
    def __getitem__(self, index):
        # Get audio data
        audio_data = self.audio_path + '\\' + self._audio_sample_name(
            index=index)
        audio, sample_rate = torchaudio.load(audio_data)
        audio = self._sample_audio_data(audio_data=audio,
                                        sample_rate=sample_rate)
        text_data = self.annotations['sentence'][index]

        return
