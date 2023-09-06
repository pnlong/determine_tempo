# README
# Phillip Long
# September 5, 2023

# Given a song, output its predicted tempo (in BPM).

# python ./determine_tempo.py nn_filepath song_filepath


# IMPORTS
##################################################
import sys
from os.path import dirname, join, exists
import torch
import torchaudio
import torchvision
import tempo_dataset as tempo_data
from tempo_neural_network import tempo_nn
# sys.argv = ("./determine_tempo.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pth", "")
##################################################


# MAIN FUNCTION
##################################################

class tempo_determiner():

    # determine device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initializer function
    def __init__(self, nn_filepath = join(dirname(__file__), "tempo_nn.pth")):
        
        # IMPORT NEURAL NETWORK
        ##################################################

        # preliminary check
        if not exists(nn_filepath):
            raise ValueError(f"Invalid nn_filepath argument: {nn_filepath} does not exist.")
        
        # import neural network, load parameters
        self.tempo_nn = tempo_nn().to(self.device)
        self.tempo_nn.load_state_dict(torch.load(nn_filepath, map_location = self.device)["state_dict"])

        ##################################################


    # determine the tempo of a given song
    def determine_tempo(self, song_filepath):

        # LOAD AUDIO, RESAMPLE, CONVERT TO MONO
        ##################################################

        # get waveform data by loading in audio
        signal, sample_rate = torchaudio.load(song_filepath, format = song_filepath.split(".")[-1]) # returns the waveform data and sample rate
        
        # register signal onto device
        signal = signal.to(self.device)

        # resample
        signal, sample_rate = tempo_data._resample_if_necessary(signal = signal, sample_rate = sample_rate, new_sample_rate = tempo_data.SAMPLE_RATE, device = self.device) # resample for consistent sample rate

        # convert from stereo to mono
        signal = tempo_data._mix_down_if_necessary(signal = signal)
        
        ##################################################


        # SPLIT SONG INTO CLIPS
        ##################################################

        # determine where to chop audio up
        start_frame, end_frame = tempo_data._trim_silence(signal = signal, sample_rate = sample_rate, window_size = 0.1) # return frames for which audible audio begins and ends
        window_size = int(tempo_data.SAMPLE_DURATION * sample_rate) # convert window size from seconds to frames
        starting_frames = tuple(range(start_frame, end_frame - window_size, int(tempo_data.STEP_SIZE * sample_rate))) # get frame numbers for which each chop starts

        # create clips, apply transforms
        inputs = torch.tensor(data = [], dtype = torch.float32).to(self.device)
        for starting_frame in starting_frames:
            clip = torch.unsqueeze(input = self._transform(signal = signal[:, starting_frame:(starting_frame + window_size)], sample_rate = sample_rate), dim = 0).to(self.device)
            inputs = torch.cat(tensors = (inputs, clip), dim = 0)

        ##################################################


        # APPLY NEURAL NETWORK, MAKE PREDICTION
        ##################################################

        # apply nn
        predictions = self.tempo_nn(inputs)
        predictions = torch.argmax(input = predictions, dim = 1, keepdim = True).view(-1) # convert to class indicies
        predictions = torch.tensor(data = list(map(lambda i: tempo_data.get_tempo(index = i), predictions)), dtype = torch.float32).view(-1) # convert predicted indicies into actual predicted tempos

        # make prediction
        predicted_tempo = float(torch.mode(input = predictions, dim = 0)[0].item())

        ##################################################

        return predicted_tempo
    

    # transform functions
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = tempo_data.SAMPLE_RATE, n_fft = tempo_data.N_FFT, n_mels = tempo_data.N_MELS).to(device) # make sure to adjust MelSpectrogram parameters such that # of mels > 224 and ceil((2 * SAMPLE_DURATION * SAMPLE_RATE) / n_fft) > 224
    normalize = torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).to(device) # normalize the image according to PyTorch docs (https://pytorch.org/vision/0.8/models.html)


    # transform a waveform into the input of the neural network
    def _transform(self, signal, sample_rate):

        # pad/crop for fixed signal duration; duration was already set in preprocessing
        signal = tempo_data._edit_duration_if_necessary(signal = signal, sample_rate = sample_rate, target_duration = tempo_data.SAMPLE_DURATION) # crop/pad if signal is too long/short

        # convert waveform to melspectrogram
        signal = self.mel_spectrogram(signal) # (single channel, # of mels, # of time samples) = (1, 128, ceil((SAMPLE_DURATION * SAMPLE_RATE) / n_fft) = 431)

        # apply autocorrelation function, (1, 128, 431) -> (1, 128, 224)        
        signal = tempo_data._melspectrogram_to_acf(signal = signal, n_lags = tempo_data.N_LAGS, device = self.device)

        # because acf is correlation, meaning its values span from -1 to 1 (inclusive), min-max normalize such that the pixel values span from 0 to 255 (also inclusive)
        signal = (signal + 1) * (255 / 2)

        # make image width satisfy PyTorch image processing requirements
        signal = torch.repeat_interleave(input = signal, repeats = (tempo_data.TORCHVISION_MIN_IMAGE_DIM // signal.size(2)) + 1, dim = 2)

        # make image height satisfy PyTorch image processing requirements, (1, 128, 224) -> (1, 256, 224)
        signal = torch.repeat_interleave(input = signal, repeats = tempo_data.IMAGE_HEIGHT // tempo_data.N_MELS, dim = 1)

        # convert from 1 channel to 3 channels (mono -> RGB); I will treat this as an image classification problem
        signal = torch.repeat_interleave(input = signal, repeats = 3, dim = 0) # (3 channels, # of mels, # of time samples) = (3, 256, 224)

        # normalize the image according to PyTorch docs (https://pytorch.org/vision/0.8/models.html)
        signal = self.normalize(signal)

        # return the signal as a transformed tensor registered to the correct device
        return signal

##################################################


# PROOF OF FUNCTION
##################################################

if __name__ == "__main__":

    song_filepath = sys.argv[2]
    nn_filepath = sys.argv[1]
    tempo_determiner = tempo_determiner(nn_filepath = nn_filepath)
    predicted_tempo = tempo_determiner.determine_tempo(song_filepath = song_filepath)
    print(f"Predicted tempo of {song_filepath}: {predicted_tempo:.0f} BPM")

##################################################