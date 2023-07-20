# Here we are using torchaudio to load audio
# For linux use backend = 'sox'
# For windows use backend = 'soundfile'

import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# Now let's create a class for our custome dataset
class AudioData(Dataset):

    def __init__(self, audio_path, annotation_path):
        super().__init__()
        self.annotation = pd.read_csv(annotation_path, sep='\t')
        self.audio_path = audio_path

    # Claculate the length of the dataset
    def __len__(self):
        return len(self.annotation)

    # Here we get the audio path to load
    def get_audio_sample_path(self, index):
        # We need the audio name
        file_name = self.annotation['path'][index]
        return file_name

    # Here we get the label from the given text
    def get_audio_label(self, index):
        text = self.annotation['sentence'][index]
        return text

    # Get the data using index
    def __getitem__(self, index):
        # Get the audio path
        audio_sample_path = self.audio_path + '\\' + self.get_audio_sample_path(
            index)
        # Get the label path
        label = self.get_audio_label(index)
        print(label)
        # Let's load both of them
        # It returns two values as tuples -> 1) wavefrom(tensor) 2) sample rate(int)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        metadata = torchaudio.info(audio_sample_path)
        return signal, sample_rate, label, metadata


# Let's test the Class
if __name__ == '__main__':
    ANNOTATION_AUDIO = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\clips'
    ANNOTATION_PATH = r'D:\Deep_Learning\Algorithm\Audio_Processing\cv-corpus-14.0-delta-2023-06-23\en\train.tsv'
    # Now pass the data to the class
    audio_data = AudioData(ANNOTATION_AUDIO, ANNOTATION_PATH)

    # Let's check how many data's are present in the dataset
    print(f'There are {len(audio_data)} samples in dataset.')
    signal, label, sample_rate, metadata = audio_data[0]
    print('Audio file: ', signal)
    print('Shape of the audio file: ', signal.shape)
    print('Audio label', label)

    # see information about the audio
    print('Audio Informations: ', metadata)

    # torchaudio.info = show metadata of the audio file
    # torchaudio.load = load any audio file
    # torchaudio.save = save any audio file