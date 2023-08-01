# README
# Phillip Long
# August 1, 2023

# Create a custom audio dataset for PyTorch with torchaudio.
# Uses songs from my music library

# python ./tempo_dataset.py labels_filepath output_filepath seconds_per_sample

# IMPORTS
##################################################
import sys
from os.path import exists
import torch
from torch.utils.data import Dataset # base dataset class to create datasets
import torchaudio
import pandas as pd
# sys.argv = ("./tempo_dataset.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_key_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv", "10")
##################################################

# TEMPO DATASET OBJECT CLASS
##################################################
class tempo_dataset(Dataset):

    def __init__(self, data_filepath, target_sample_rate, sample_duration, device, transformation):

        # import labelled data file
        self.data = pd.read_csv(data_filepath, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
        self.data = self.data[self.data["path"].apply(lambda path: exists(path))] # remove files that do not exist
        self.data = self.data[~pd.isna(self.data["tempo"])] # remove na values
        self.data.reset_index(drop = True) # reset indicies

        # import constants
        self.target_sample_rate = target_sample_rate
        self.n_samples = int(sample_duration * self.target_sample_rate)
        self.device = device

        # import torch audio transformation
        self.transformation = transformation.to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get waveform data by loading in audio
        signal, sr = torchaudio.load(self.data.at[index, "path"], format = "mp3") # returns the waveform data and sample rate

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

        return signal, self.data.at[index, "tempo"] # returns the transformed signal and the actual BPM

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

##################################################

# TEST IF DATASET OBJECT WORKS
##################################################
# if __name__ == "__main__" only runs the code inside the if statement when the program is run directly by the Python interpreter.
# The code inside the if statement is not executed when the file's code is imported as a module.
if __name__ == "__main__":

    # constants
    LABELS_FILEPATH = sys.argv[1]
    OUTPUT_FILEPATH = sys.argv[2]
    SAMPLE_DURATION = float(sys.argv[3]) # in seconds
    SAMPLE_RATE = 44100 // 2

    # determine what device to run things on
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 1024 // 2, n_mels = 64)

    tempo_data = tempo_dataset(data_filepath = LABELS_FILEPATH, target_sample_rate = SAMPLE_RATE, sample_duration = SAMPLE_DURATION, device = device, transformation = mel_spectrogram)

    print(f"There are {len(tempo_data)} samples in the dataset.")

    signal, label = tempo_data[0]
##################################################