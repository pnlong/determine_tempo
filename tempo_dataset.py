# README
# Phillip Long
# August 1, 2023

# Create a custom audio dataset for PyTorch with torchaudio.
# Uses songs from my music library.

# python ./tempo_dataset.py labels_filepath output_filepath audio_dir


# IMPORTS
##################################################
import sys
from os.path import exists, join, dirname
from os import makedirs, remove
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset # base dataset class to create datasets
import torchaudio
import torchvision.transforms
import pandas as pd
from statsmodels.tsa.stattools import acf
# sys.argv = ("./tempo_dataset.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_key_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv", "/Volumes/Seagate/artificial_dj_data/tempo_data")
##################################################


# CONSTANTS
##################################################
SAMPLE_RATE = 44100 // 2
SAMPLE_DURATION = 10.0 # in seconds
STEP_SIZE = SAMPLE_DURATION / 2 # in seconds, the amount of time between the start of each .wav file
TEMPO_RANGE = tuple(85 * i for i in (1, 2)) # set the minimum tempo, this will create a range of non-duplicate tempos (exclusive, inclusive)
TEMPO_MAPPINGS = tuple(range(TEMPO_RANGE[0], TEMPO_RANGE[1] + 1))
TORCHVISION_MIN_IMAGE_DIM = 224 # 224 is the minimum image width for PyTorch image processing, for waveform to melspectrogram transformation
IMAGE_HEIGHT = 256 # height of the resulting image after transforms are applied
N_MELS = IMAGE_HEIGHT // 2 # for waveform to melspectrogram transformation
# determine constants for melspectrogram and acf
N_BINS_PER_BEAT_AT_MAX_TEMPO = 20 # cannot exceed (60 / TEMPO_RANGE[1]) * SAMPLE_RATE; increase to increase resolution of acf
BIN_LENGTH = max((60 / TEMPO_RANGE[1]) / N_BINS_PER_BEAT_AT_MAX_TEMPO, 1 / SAMPLE_RATE) # length (in seconds) of each time-bin in the mel spectrogram; edit the term outside the parentheses (directly related to n_lags)
ACF_EXTENDS_N_TIMES_PAST_BEAT_AT_MIN_TEMPO = 1.5 # at the slowest tempo, the ACF should cover one beat plus some certain amount; this value is how many times longer the ACF should extend than the beat at the slowest tempo
N_LAGS = int(((60 / TEMPO_RANGE[0]) * ACF_EXTENDS_N_TIMES_PAST_BEAT_AT_MIN_TEMPO) / BIN_LENGTH) + 1 # number of lags for autocorrelation function
N_FFT = int(2 * BIN_LENGTH * SAMPLE_RATE) # enter in number of time-bins before applying autocorrelation function; previously min(1024, (2 * SAMPLE_DURATION * SAMPLE_RATE) // TORCHVISION_MIN_IMAGE_DIM); number of samples in each bin on the mel spectrogram
SET_TYPES = {"train": 0.7, "validate": 0.2, "test": 0.1} # train-validation-test fractions
##################################################


# TEMPO DATASET OBJECT CLASS
##################################################

class tempo_dataset(Dataset):

    def __init__(self, labels_filepath, set_type, device, use_pseudo_replicates = True):
        # set_type can take on one of three values: ("train", "validate", "test")

        # import labelled data file, preprocess
        # it is assumed that the data are mono wav files
        self.data = pd.read_csv(labels_filepath, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
        self.data = self.data[self.data["path"].apply(lambda path: exists(path))] # remove files that do not exist
        self.data = self.data[~pd.isna(self.data["tempo"])] # remove na values
        if not use_pseudo_replicates: # if no pseudo-replicates, transform self.data once more
            self.data = self.data.groupby(["title", "artist", "key", "path_origin"]).sample(n = 1, replace = False, random_state = 0, ignore_index = True) # randomly pick a sample from each song
            self.data = self.data.reset_index(drop = True) # reset indicies

        # partition into the train, validation, or test dataset
        self.data = self.data.sample(frac = 1, replace = False, random_state = 0, ignore_index = True) # shuffle data
        set_type = "" if set_type.lower()[:3] not in map(lambda key: key[:3], SET_TYPES.keys()) else str(*(key for key in SET_TYPES.keys() if key[:3] == set_type.lower()[:3])) # get the SET_TYPES key closest to set_type
        self.data = self.data.iloc[_partition(n = len(self.data))[set_type]].reset_index(drop = True) # extract range depending on set_type, also reset indicies
        
        # import constants
        self.device = device

        # instantiate transformation functions  
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = N_FFT, n_mels = N_MELS).to(self.device) # make sure to adjust MelSpectrogram parameters such that # of mels > 224 and ceil((2 * SAMPLE_DURATION * SAMPLE_RATE) / n_fft) > 224
        self.normalize = torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).to(self.device) # normalize the image according to PyTorch docs (https://pytorch.org/vision/0.8/models.html)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get waveform data by loading in audio
        signal, sample_rate = torchaudio.load(self.data.at[index, "path"], format = "wav") # returns the waveform data and sample rate
        # register signal onto device (gpu [cuda] or cpu)
        signal = signal.to(self.device)
        # apply transformations
        signal = self._transform(signal = signal, sample_rate = sample_rate)
        # return the transformed signal and the actual BPM
        return signal, torch.tensor([self.data.at[index, "tempo"]], dtype = torch.float32)
    
    # get info (title, artist, original filepath) of a file given its index; return as dictionary
    def get_info(self, index):
        return self.data.loc[index, ["title", "artist", "path_origin", "path", "tempo", "key"]].to_dict()
    
    # transform a waveform into whatever will be used to train a neural network
    def _transform(self, signal, sample_rate):

        # resample; sample_rate was already set in preprocessing
        # signal, sample_rate = _resample_if_necessary(signal = signal, sample_rate = sample_rate, new_sample_rate = SAMPLE_RATE, device = self.device) # resample for consistent sample rate
        
        # convert from stereo to mono; already done in preprocessing
        # signal = _mix_down_if_necessary(signal = signal)

        # pad/crop for fixed signal duration; duration was already set in preprocessing
        # signal = _edit_duration_if_necessary(signal = signal, sample_rate = sample_rate, target_duration = SAMPLE_DURATION) # crop/pad if signal is too long/short

        # convert waveform to melspectrogram
        signal = self.mel_spectrogram(signal) # (single channel, # of mels, # of time samples) = (1, 128, ceil((SAMPLE_DURATION * SAMPLE_RATE) / n_fft) = 431)

        # apply autocorrelation function, (1, 128, 431) -> (1, 128, 224)        
        signal = _melspectrogram_to_acf(signal = signal, n_lags = N_LAGS, device = self.device)

        # because acf is correlation, meaning its values span from -1 to 1 (inclusive), min-max normalize such that the pixel values span from 0 to 255 (also inclusive)
        signal = (signal + 1) * (255 / 2)

        # make image width satisfy PyTorch image processing requirements
        signal = torch.repeat_interleave(input = signal, repeats = (TORCHVISION_MIN_IMAGE_DIM // signal.size(2)) + 1, dim = 2)

        # make image height satisfy PyTorch image processing requirements, (1, 128, 224) -> (1, 256, 224)
        signal = torch.repeat_interleave(input = signal, repeats = IMAGE_HEIGHT // N_MELS, dim = 1)

        # convert from 1 channel to 3 channels (mono -> RGB); I will treat this as an image classification problem
        signal = torch.repeat_interleave(input = signal, repeats = 3, dim = 0) # (3 channels, # of mels, # of time samples) = (3, 256, 224)

        # normalize the image according to PyTorch docs (https://pytorch.org/vision/0.8/models.html)
        signal = self.normalize(signal)

        # return the signal as a transformed tensor registered to the correct device
        return signal

    # sample n_predictions random rows from data, return a tensor of the audios and a tensor of the labels
    # def sample(self, n_predictions):
    #     inputs_targets = [self.__getitem__(index = i) for i in self.data.sample(n = n_predictions, replace = False, ignore_index = False).index]
    #     inputs = torch.cat([torch.unsqueeze(input = input_target[0], dim = 0) for input_target in inputs_targets], dim = 0).to(self.device) # tempo_nn expects (batch_size, num_channels, frequency, time) [4-dimensions], so we add the batch size dimension here with unsqueeze()
    #     targets = torch.cat([input_target[1] for input_target in inputs_targets], dim = 0).view(n_predictions, 1).to(self.device) # note that I register the inputs and targets tensors to whatever device we are using
    #     del inputs_targets
    #     return inputs, targets

##################################################


# HELPER FUNCTIONS
##################################################

# partition dataset into training, validation, and test sets
def _partition(n, set_types = SET_TYPES):
    set_types_values = [int(i.item()) for i in (torch.cumsum(input = torch.Tensor([0,] + list(set_types.values())), dim = 0) * n)] # get indicies for start of each new dataset type
    set_types_values = [range(set_types_values[i - 1], set_types_values[i]) for i in range(1, len(set_types_values))] # create ranges from previously created indicies
    set_types = dict(zip(set_types.keys(), set_types_values)) # create new set types dictionary
    set_types[""] = range(n) # create instance where no set type is named, so return all values
    return set_types

# resampler
def _resample_if_necessary(signal, sample_rate, new_sample_rate, device):
        if sample_rate != new_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq = sample_rate, new_freq = new_sample_rate).to(device)
            signal = resampler(signal)
        return signal, new_sample_rate

# convert from stereo to mono
def _mix_down_if_necessary(signal):
    if signal.size(0) > 1: # signal.size(0) = # of channels; if # of channels is more than one, it is stereo, so convert to mono
        signal = torch.mean(signal, dim = 0, keepdim = True)
    return signal

# crop/pad if waveform is too long/short
def _edit_duration_if_necessary(signal, sample_rate, target_duration):
    n = int(target_duration * sample_rate) # n = desired signal length in # of samples; convert from seconds to # of samples
    if signal.size(1) > n: # crop if too long
        signal = signal[:, :n]
    elif signal.size(1) < n: # zero pad if too short
        last_dim_padding = (0, n - signal.shape[1])
        signal = torch.nn.functional.pad(signal, pad = last_dim_padding, value = 0)
    return signal

# autocorrelation function, takes the Mel Spectrogram as input (3-dimensional tensor) and returns a 3-d tensor with an acf applied to each mel (dim = 1)
def _melspectrogram_to_acf(signal, n_lags, device):
    signal_acf = torch.empty(size = signal.shape[:2] + (n_lags,), dtype = signal.dtype) # define empty tensor to store values in and return
    for n_mel in range(signal.size(1)): # compute acf at each mel bin
        signal_acf[0, n_mel] = torch.tensor(data = acf(x = signal[0, n_mel].tolist(), nlags = n_lags - 1), dtype = signal.dtype) # compute acf, convert to tensor, store in output
    return signal_acf.to(device) # re-register with device since tensor was coverted to numpy and back

# given a mono signal, return the sample #s for which the song begins and ends (trimming silence)
def _trim_silence(signal, sample_rate, window_size = 0.1): # window_size = size of rolling window (in SECONDS)
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
    end_frame = starting_frames[len(is_silence) - is_silence[::-1].index(False) - 1] if sum(is_silence) != len(is_silence) else 0 # get ending from of audible audio
    return start_frame, end_frame

# fix duplicate tempos (e.g. 60 BPM is equivalent to 120 BPM)
def fix_duplicate_tempo(tempo):
    tempo = float(tempo) # convert to float in case
    while tempo <= TEMPO_RANGE[0]:
        tempo *= 2
    while tempo > TEMPO_RANGE[1]:
        tempo /= 2
    return tempo

##################################################


# ACCESSOR METHODS
##################################################

# TEMPO
# get the tempo index from tempo
def get_tempo_index(tempo):
    tempo = fix_duplicate_tempo(tempo = tempo)
    # return TEMPO_MAPPINGS.index(int(round(fix_duplicate_tempo(tempo = tempo)))) # works when TEMPO_MAPPINGS are whole numbers
    return min(range(len(TEMPO_MAPPINGS)), key = lambda i: abs(tempo - TEMPO_MAPPINGS[i])) # more versatile

# get the tempo from the tempo index
def get_tempo(index):
    return float(TEMPO_MAPPINGS[index])

##################################################


# if __name__ == "__main__" only runs the code inside the if statement when the program is run directly by the Python interpreter.
# The code inside the if statement is not executed when the file's code is imported as a module.
if __name__ == "__main__":

    # CONSTANTS
    ##################################################
    LABELS_FILEPATH = sys.argv[1]
    OUTPUT_FILEPATH = sys.argv[2]
    AUDIO_DIR = sys.argv[3]
    ##################################################


    # CREATE AND PREPROCESS WAV FILE CHOPS FROM FULL SONGS
    ##################################################

    # create audio output directory and output_filepath
    if not exists(AUDIO_DIR): 
        makedirs(AUDIO_DIR)
    if not exists(dirname(OUTPUT_FILEPATH)):
        makedirs(dirname(OUTPUT_FILEPATH))   

    # clear AUDIO_DIR
    for filepath in tqdm(glob(join(AUDIO_DIR, "*")), desc = f"Clearing files from {AUDIO_DIR}"):
        remove(filepath)
    
    # determine what device to run things on
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    
    # load in labels
    data = pd.read_csv(LABELS_FILEPATH, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
    data = data[data["path"].apply(lambda path: exists(path))] # remove files that do not exist
    data = data[(~pd.isna(data["tempo"])) & (data["tempo"] > 0.0)] # remove NA and unclassified tempos
    data = data.reset_index(drop = True) # reset indicies
    
    # loop through songs and create .wav files
    origin_filepaths, output_filepaths, tempos = [], [], []
    for i in tqdm(data.index, desc = "Chopping up songs into WAV files"): # start from start index

        # preprocess audio
        try: # try to import the file
            signal, sample_rate = torchaudio.load(data.at[i, "path"], format = "mp3") # load in the audio file
        except RuntimeError:
            continue
        signal = signal.to(device) # register signal onto device (gpu [cuda] or cpu)
        signal, sample_rate = _resample_if_necessary(signal = signal, sample_rate = sample_rate, new_sample_rate = SAMPLE_RATE, device = device) # resample for consistent sample rate
        signal = _mix_down_if_necessary(signal = signal) # if there are multiple channels, convert from stereo to mono

        # chop audio into many wav files
        start_frame, end_frame = _trim_silence(signal = signal, sample_rate = sample_rate, window_size = 0.1) # return frames for which audible audio begins and ends
        window_size = int(SAMPLE_DURATION * sample_rate) # convert window size from seconds to frames
        starting_frames = tuple(range(start_frame, end_frame - window_size, int(STEP_SIZE * sample_rate))) # get frame numbers for which each chop starts
        origin_filepath = data.at[i, "path"] # set original filepath
        tempo = fix_duplicate_tempo(tempo = data.at[i, "tempo"]) # set the tempo
        for j, starting_frame in enumerate(starting_frames):
            output_filepath = join(AUDIO_DIR, f"{i}_{j}.wav") # create filepath
            torchaudio.save(output_filepath, signal[:, starting_frame:(starting_frame + window_size)], sample_rate = sample_rate, format = "wav") # save chop as .wav file
            origin_filepaths.append(origin_filepath) # add original filepath to origin_filepaths
            output_filepaths.append(output_filepath) # add filepath to output_filepaths
            tempos.append(tempo) # add tempo to tempos
        
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

    # instantiate tempo dataset
    tempo_data = tempo_dataset(labels_filepath = OUTPUT_FILEPATH, set_type = "", device = device)

    # test len() functionality
    print(f"There are {len(tempo_data)} samples in the dataset.")

    # test __getitem__ functionality
    signal, label = tempo_data[0]

    # test get_info() functionality
    print(f"The artist of the 0th sample is {tempo_data.get_info(0)['artist']}.")

    ##################################################


    # CODE FOR FIXING DUPLICATE TEMPOS IN HINDSIGHT
    ##################################################

    # tempo_data = pd.read_csv(OUTPUT_FILEPATH, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
    # tempo_data["tempo"] = tempo_data["tempo"].apply(fix_duplicate_tempo)
    # tempo_data.to_csv(OUTPUT_FILEPATH, sep = "\t", header = True, index = False, na_rep = "NA") # write output

    ##################################################


    # MAKE PLOT TO SHOW PREPROCESSING TRANSFORMS
    ##################################################

    def make_preprocessing_plot(audio_filepath, output_filepath):
        # imports
        import matplotlib.pyplot as plt
        from numpy import arange, linspace

        # some constants
        device = "cpu" # perform computation on cpu

        # load in audio file
        signal, sample_rate = torchaudio.load(audio_filepath, format = "wav") # returns the waveform data and sample rate
        # register signal onto device (gpu [cuda] or cpu)
        signal = signal.to(device)
        # resample; sample_rate was already set in preprocessing
        signal, sample_rate = _resample_if_necessary(signal = signal, sample_rate = sample_rate, new_sample_rate = SAMPLE_RATE, device = device) # resample for consistent sample rate
        # convert from stereo to mono; already done in preprocessing
        signal = _mix_down_if_necessary(signal = signal)
        # pad/crop for fixed signal duration; duration was already set in preprocessing
        signal = _edit_duration_if_necessary(signal = signal, sample_rate = sample_rate, target_duration = SAMPLE_DURATION) # crop/pad if signal is too long/short
        # convert waveform to melspectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = N_FFT, n_mels = N_MELS).to(device) # make sure to adjust MelSpectrogram parameters such that # of mels > 224 and ceil((2 * SAMPLE_DURATION * SAMPLE_RATE) / n_fft) > 224
        signal = mel_spectrogram(signal)

        # create plot
        fig, (ms_plot, acf_plot) = plt.subplots(nrows = 1, ncols = 2, layout = "constrained", figsize = (12, 8))
        fig.suptitle(f"Audio Preprocessing")

        # plot melspectrogam
        ms_plot_temp = ms_plot.imshow(signal[0], aspect = "auto", origin = "lower", cmap = "plasma")
        fig.colorbar(ms_plot_temp, ax = ms_plot, label = "Amplitude", location = "top")
        ms_plot.set_xticks(ticks = arange(start = 0, stop = signal.size(2) + 1, step = signal.size(2) / 10), labels = arange(start = 0, stop = 10 + 1, step = 1))
        ms_plot.set_xlabel("Time (seconds)")
        ms_plot.set_ylabel("Frequency (mels)")
        ms_plot.set_title("Melspectrogram")

        # apply autocorrelation function      
        signal = _melspectrogram_to_acf(signal = signal, n_lags = N_LAGS, device = device)

        # plot autocorrelation function
        acf_plot_temp = acf_plot.imshow(signal[0], aspect = "auto", origin = "lower", cmap = "Greys")
        fig.colorbar(acf_plot_temp, ax = acf_plot, label = "Correlation", location = "top")
        last_label = round((N_LAGS * BIN_LENGTH) * 10) / 10
        acf_plot.set_xticks(ticks = linspace(start = 0, stop = (signal.size(2) * last_label) / (N_LAGS * BIN_LENGTH), num = int(last_label * 10) + 1), labels = (f"{num:.1f}" for num in arange(start = 0, stop = last_label + 0.01, step = 0.1)))
        acf_plot.set_xlabel("Time (seconds)")
        acf_plot.sharey(other = ms_plot)
        acf_plot.set_title("Autocorrelation Function")

        # save figure
        fig.savefig(output_filepath, dpi = 240)
        print(f"Saved plot to {output_filepath}.")

    ##################################################
