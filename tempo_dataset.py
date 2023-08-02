# README
# Phillip Long
# August 1, 2023

# Create a custom audio dataset for PyTorch with torchaudio.
# Uses songs from my music library

# python ./tempo_dataset.py labels_filepath output_filepath audio_dir seconds_per_sample


# IMPORTS
##################################################
import sys
from os.path import exists, join
from os import makedirs
import torch
from torch.utils.data import Dataset # base dataset class to create datasets
import torchaudio
import pandas as pd
from tqdm import tqdm
# sys.argv = ("./tempo_dataset.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_key_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv", "/Volumes/Seagate/artificial_dj_data/tempo_data", "10")
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

# if __name__ == "__main__" only runs the code inside the if statement when the program is run directly by the Python interpreter.
# The code inside the if statement is not executed when the file's code is imported as a module.
if __name__ == "__main__":

    # constants
    LABELS_FILEPATH = sys.argv[1]
    OUTPUT_FILEPATH = sys.argv[2]
    AUDIO_DIR = sys.argv[3]
    SAMPLE_DURATION = float(sys.argv[4]) # in seconds
    SAMPLE_RATE = 44100 // 2
    STEP_SIZE = SAMPLE_DURATION / 2 # in seconds, the amount of time between each .wav file


    # CREATE AND PREPROCESS WAV FILE CHOPS FROM FULL SONGS
    ##################################################
    # create audio output directory if it is not yet created
    if not exists(AUDIO_DIR): 
        makedirs(AUDIO_DIR)
    
    # determine what device to run things on
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    
    # load in labels
    data = pd.read_csv(LABELS_FILEPATH, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
    data = data[data["path"].apply(lambda path: exists(path))] # remove files that do not exist
    data = data[(~pd.isna(data["tempo"])) & (data["tempo"] > 0.0)] # remove NA and unclassified tempos
    data = data.reset_index(drop = True) # reset indicies

    # some helpful functions for preprocessing audio
    def resample_if_necessary(signal, sample_rate): # resample
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq = sample_rate, new_freq = SAMPLE_RATE).to(device)
            signal = resampler(signal)
        return signal, SAMPLE_RATE # return signal and new sample rate
    def mix_down_if_necessary(signal): # convert from stereo to mono
        if signal.shape[0] > 1: # signal.shape[0] = # of channels; if # of channels is more than one, it is stereo, so convert to mono
            signal = torch.mean(signal, dim = 0, keepdim = True)
        return signal
    def trim_silence(signal, sample_rate, window_size = 0.1, threshold = 0): # given a flattened mono signal, return the sample #s for which the song begins and ends (trimming silence)
        # window_size = size of rolling window (in SECONDS)
        signal = torch.abs(input = signal) # make all values positive
        window_size = int(sample_rate * window_size) # convert window size from seconds to # of samples
        starting_frames = tuple(range(0, len(signal), window_size)) # determine starting frames
        is_silence = [bool(torch.mean(input = signal[i:(i + window_size)]) < threshold) for i in starting_frames] # slide window over audio and look for silence
        start_frame = starting_frames[is_silence.index(False)] if sum(is_silence) != len(is_silence) else 0 # get starting frame of audible audio
        end_frame = starting_frames[len(is_silence) - is_silence[::-1].index(False)] if sum(is_silence) != len(is_silence) else 0 # get ending from of audible audio
        return start_frame, end_frame
    
    # loop through songs and create .wav files
    origin_filepaths, output_filepaths, tempos = [], [], []
    for i in tqdm(data.index):

        # preprocess audio
        signal, sample_rate = torchaudio.load(data.at[i, "path"], format = "mp3") # load in the audio file
        signal = signal.to(device) # register signal onto device (gpu [cuda] or cpu)
        signal, sample_rate = resample_if_necessary(signal = signal, sample_rate = sample_rate) # resample for consistent sample rate
        signal = mix_down_if_necessary(signal = signal) # if there are multiple channels, convert from stereo to mono
        signal = torch.flatten(input = signal) # flatten signal into 1D tensor

        # chop audio into many wav files
        start_frame, end_frame = trim_silence(signal = signal, sample_rate = sample_rate, window_size = 0.1, threshold = 0.01) # return frames for which audible audio begins and ends
        window_size = int(SAMPLE_DURATION * sample_rate) # convert window size from seconds to frames
        starting_frames = tuple(range(start_frame, end_frame - window_size, int(STEP_SIZE * sample_rate))) # get frame numbers for which each chop starts
        for chop_index, j in enumerate(starting_frames):
            path = join(AUDIO_DIR, f"{i}_{chop_index}.wav") # create filepath
            torchaudio.save(path = path,
                            waveform = signal[j:(j + window_size)].view(1, window_size),
                            sample_rate = sample_rate,
                            format = "wav") # save chop as .wav file
            origin_filepaths.append(data.at[i, "path"])
            output_filepaths.append(path) # add filepath to filepaths
            tempos.append(data.at[i, "tempo"]) # add tempo to tempos

    # write to OUTPUT_FILEPATH
    data = data.rename(columns = {"path": "path_origin"}).drop(columns = ["tempo"]) # rename path column in the original dataframe
    tempo_data = pd.DataFrame(data = {"path_origin": origin_filepaths, "path": output_filepaths, "tempo": tempos})
    tempo_data = pd.merge(tempo_data, data, on = "path_origin", how = "left").reset_index(drop = True)
    tempo_data = [["title", "artist", "album", "genre", "key", "path_origin", "path", "tempo"]]
    print(f"\nWriting output to {OUTPUT_FILEPATH}.")
    tempo_data.to_csv(OUTPUT_FILEPATH, sep = "\t", header = True, index = False, na_rep = "NA")
    ##################################################


    # TEST DATASET OBJECT
    ##################################################

    # instantiate mel spectrogram transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 1024 // 2, n_mels = 64)

    # instantiate tempo dataset
    tempo_data = tempo_dataset(data_filepath = LABELS_FILEPATH, target_sample_rate = SAMPLE_RATE, sample_duration = SAMPLE_DURATION, device = device, transformation = mel_spectrogram)

    # test len() functionality
    print(f"There are {len(tempo_data)} samples in the dataset.")

    # test __getitem__ functionality
    signal, label = tempo_data[0]
    
    ##################################################