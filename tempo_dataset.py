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

        # import labelled data file, preprocess
        # it is assumed that the data are mono wav files
        self.data = pd.read_csv(data_filepath, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
        self.data = self.data[self.data["path"].apply(lambda path: exists(path))] # remove files that do not exist
        self.data = self.data[~pd.isna(self.data["tempo"])] # remove na values
        self.data.reset_index(drop = True) # reset indicies

        # import constants
        self.target_sample_rate = target_sample_rate
        self.sample_duration = sample_duration
        self.device = device

        # import torch audio transformation(s)
        self.transformation = transformation.to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get waveform data by loading in audio
        signal, sample_rate = torchaudio.load(self.data.at[index, "path"], format = "wav") # returns the waveform data and sample rate
        # register signal onto device (gpu [cuda] or cpu)
        signal = signal.to(self.device)
        # resample
        signal = self._resample_if_necessary(signal = signal, sample_rate = sample_rate) # resample for consistent sample rate
        # pad/crop for fixed signal duration
        signal = self._edit_duration_if_necessary(signal = signal) # crop/pad if signal is too long/short
        # apply transformations
        signal = self.transformation(signal) # convert waveform to melspectrogram

        return signal, self.data.at[index, "tempo"] # returns the transformed signal and the actual BPM
    
    def get_info(self, index): # get info (title, artist, original filepath) of a file given its index; return as dictionary
        return self.data.loc[i, ["title", "artist", "key", "path_origin", "path"]].to_dict()

    def _resample_if_necessary(self, signal, sample_rate): # resample
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq = sample_rate, new_freq = self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    
    def _edit_duration_if_necessary(self, signal): # crop/pad if waveform is too long/short
        n = int(self.sample_duration * self.target_sample_rate) # n = desired signal length in # of samples; convert from seconds to # of samples
        if signal.shape[1] > n: # crop if too long
            signal = signal[:, :n]
        elif signal.shape[1] < n: # zero pad if too short
            last_dim_padding = (0, n - signal.shape[1])
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

    # resample audio function
    def resample_if_necessary(signal, sample_rate):
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq = sample_rate, new_freq = SAMPLE_RATE).to(device)
            signal = resampler(signal)
        return signal, SAMPLE_RATE # return signal and new sample rate
    
    # convert from stereo to mono
    def mix_down_if_necessary(signal):
        if signal.shape[0] > 1: # signal.shape[0] = # of channels; if # of channels is more than one, it is stereo, so convert to mono
            signal = torch.mean(signal, dim = 0, keepdim = True)
        return signal
    
    # given a mono signal, return the sample #s for which the song begins and ends (trimming silence)
    def trim_silence(signal, sample_rate, window_size = 0.1): # window_size = size of rolling window (in SECONDS)
        # preprocess signal
        signal = torch.flatten(input = signal) # flatten signal into 1D tensor
        signal = torch.abs(input = signal) # make all values positive
        # parse signal with a sliding window
        window_size = int(sample_rate * window_size) # convert window size from seconds to # of samples
        starting_frames = tuple(range(0, len(signal), window_size)) # determine starting frames
        is_silence = [torch.mean(input = signal[i:(i + window_size)]).item() for i in starting_frames] # slide window over audio and get average level for each window
        # determine a threshold to cutoff audio
        threshold = max(is_silence) * 1e-4 # determine a threshold, ADJUST THIS VALUE TO ADJUST THE CUTOFF THRESHOLD
        is_silence = [x < threshold for x in is_silence] # determine which windows are below the threshold
        start_frame = starting_frames[is_silence.index(False)] if sum(is_silence) != len(is_silence) else 0 # get starting frame of audible audio
        end_frame = starting_frames[len(is_silence) - is_silence[::-1].index(False)] if sum(is_silence) != len(is_silence) else 0 # get ending from of audible audio
        return start_frame, end_frame
    
    # loop through songs and create .wav files
    origin_filepaths, output_filepaths, tempos = [], [], []
    for i in tqdm(data.index, desc = "Chopping up songs into WAV files"):

        # preprocess audio
        signal, sample_rate = torchaudio.load(data.at[i, "path"], format = "mp3") # load in the audio file
        signal = signal.to(device) # register signal onto device (gpu [cuda] or cpu)
        signal, sample_rate = resample_if_necessary(signal = signal, sample_rate = sample_rate) # resample for consistent sample rate
        signal = mix_down_if_necessary(signal = signal) # if there are multiple channels, convert from stereo to mono

        # chop audio into many wav files
        start_frame, end_frame = trim_silence(signal = signal, sample_rate = sample_rate, window_size = 0.1) # return frames for which audible audio begins and ends
        window_size = int(SAMPLE_DURATION * sample_rate) # convert window size from seconds to frames
        starting_frames = tuple(range(start_frame, end_frame - window_size, int(STEP_SIZE * sample_rate))) # get frame numbers for which each chop starts
        for j, starting_frame in enumerate(starting_frames):
            path = join(AUDIO_DIR, f"{i}_{j}.wav") # create filepath
            torchaudio.save(path, signal[:, starting_frame:(starting_frame + window_size)], sample_rate = sample_rate, format = "wav") # save chop as .wav file
            origin_filepaths.append(data.at[i, "path"]) # add original filepath to origin_filepaths
            output_filepaths.append(path) # add filepath to output_filepaths
            tempos.append(data.at[i, "tempo"]) # add tempo to tempos

    # write to OUTPUT_FILEPATH
    data = data.rename(columns = {"path": "path_origin"}).drop(columns = ["tempo"]) # rename path column in the original dataframe
    tempo_data = pd.DataFrame(data = {"path_origin": origin_filepaths, "path": output_filepaths, "tempo": tempos}) # create tempo_data dataframe
    tempo_data = pd.merge(tempo_data, data, on = "path_origin", how = "left").reset_index(drop = True) # left-join tempo_data and data
    tempo_data = tempo_data[["title", "artist", "key", "path_origin", "path", "tempo"]] # select columns
    # most of the information in tempo_data is merely to help me locate a file if it causes problem; in an ideal world, I should be able to ignore it
    print(f"\nWriting output to {OUTPUT_FILEPATH}.")
    tempo_data.to_csv(OUTPUT_FILEPATH, sep = "\t", header = True, index = False, na_rep = "NA") # write output
    ##################################################


    # TEST DATASET OBJECT
    ##################################################

    # instantiate mel spectrogram transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 1024 // 2, n_mels = 64)

    # instantiate tempo dataset
    tempo_data = tempo_dataset(data_filepath = OUTPUT_FILEPATH, target_sample_rate = SAMPLE_RATE, sample_duration = SAMPLE_DURATION, device = device, transformation = mel_spectrogram)

    # test len() functionality
    print(f"There are {len(tempo_data)} samples in the dataset.")

    # test __getitem__ functionality
    signal, label = tempo_data[0]

    # test get_info() functionality
    print(f"The artist of the 0th sample is {tempo_data.get_info(0)['artist']}.")

    ##################################################