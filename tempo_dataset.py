# README
# Phillip Long
# August 1, 2023

# Create a custom audio dataset for PyTorch with torchaudio.
# Uses songs from my music library

# python ./tempo_dataset.py



from os.path import join
import torch
from torch.utils.data import Dataset # base dataset class to create datasets
import torchaudio
import pandas as pd


class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, target_sample_rate, n_samples, device, transformation):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.n_samples = n_samples
        self.device = device
        self.transformation = transformation.to(self.device)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)

        # get waveform data by loading in audio
        signal, sr = torchaudio.load(audio_sample_path) # returns the waveform data and sample rate

        # register signal onto device (gpu [cuda] or cpu)
        signal = signal.to(self.device)

        # resample
        signal = self._resample_if_necessary(signal, sample_rate = sr) # resample for consistent sample rate

        # downmix if necessary (stereo -> mono)
        signal = self._mix_down_if_necessary(signal) # if there are multiple channels, convert from stereo to mono

        # pad/crop for fixed signal duration
        signal = self._edit_duration_if_necessary(signal) # crop/pad if signal is too long/short

        # apply transformations
        signal = self.transformation(signal) # convert waveform to melspectrogram
        return signal, label

    def _get_audio_sample_path(self, index): # determine filepath for the audio file at a given index
        fold = "fold" + str(self.annotations.at[index, "fold"])
        path = join(self.audio_dir, fold, self.annotations.at[index, "slice_file_name"])
        return path

    def _get_audio_sample_label(self, index): # get target (label) for a given index
        return self.annotations.at[index, "classID"]

    def _resample_if_necessary(self, signal, sample_rate): # resample
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq = sample_rate, new_freq = self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal): # convert from stereo to mono
        if signal.shape[0] > 1: # signal.shape[0] = # of channels; if # of channels is more than one, it is stereo, so convert to mono
            signal = torch.mean(signal, dim = 0, keepdim = True)
        return signal
    
    def _edit_duration_if_necessary(self, signal): # crop/pad if waveform is too long/short
        if signal.shape[1] > self.n_samples: # crop if too long
            signal = signal[:, :self.n_samples]
        elif signal.shape[1] < self.n_samples: # zero pad if too short
            last_dim_padding = (0, self.n_samples - signal.shape[1])
            signal = torch.nn.functional.pad(signal, pad = last_dim_padding, value = 0)
        return signal


if __name__ == "__main__":

    # constants
    ANNOTATIONS_FILE = "/Volumes/Seagate/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/Volumes/Seagate/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    N_SAMPLES = 22050

    # determine what device to run things on
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 1024 // 2, n_mels = 64)

    usd = UrbanSoundDataset(annotations_file = ANNOTATIONS_FILE,
                            audio_dir = AUDIO_DIR,
                            target_sample_rate = SAMPLE_RATE,
                            n_samples = N_SAMPLES,
                            device = device,
                            transformation = mel_spectrogram
                            )

    print(f"There are {len(usd)} samples in the dataset.")

    signal, label = usd[0]