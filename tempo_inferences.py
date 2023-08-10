# README
# Phillip Long
# August 1, 2023

# Uses a neural network to make predictions of songs' tempos.

# python ./tempo_inferences.py labels_filepath nn_filepath n_predictions
# python /dfs7/adl/pnlong/artificial_dj/determine_tempo/tempo_inferences.py "/dfs7/adl/pnlong/artificial_dj/data/tempo_data.cluster.tsv" "/dfs7/adl/pnlong/artificial_dj/data/tempo_nn.pth" "100"


# IMPORTS
##################################################
import sys
from numpy import mean, percentile
from os.path import join, dirname
import matplotlib.pyplot as plt
import torch
import torchaudio
from tempo_dataset import tempo_dataset, SAMPLE_RATE, SAMPLE_DURATION 
from tempo_neural_network import tempo_nn # import neural network class
# sys.argv = ("./tempo_inferences.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pth", "20")
##################################################


# CONSTANTS
##################################################
LABELS_FILEPATH = sys.argv[1]
NN_FILEPATH = sys.argv[2]
##################################################


# RELOAD MODEL AND MAKE PREDICTIONS
##################################################

# I want to do all the predictions on CPU, GPU seems a bit much
device = "cpu"

# load back the model
tempo_nn = tempo_nn()
state_dict = torch.load(NN_FILEPATH, map_location = device)
tempo_nn.load_state_dict(state_dict["state_dict"])
print("Imported neural network parameters.")

# instantiate our dataset object
tempo_data = tempo_dataset(labels_filepath = LABELS_FILEPATH,
                           set_type = "validate",
                           target_sample_rate = SAMPLE_RATE,
                           sample_duration = SAMPLE_DURATION,
                           device = device,
                           transformation = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 1024 // 2, n_mels = 64)
                           )

# get a sample from the urban sound dataset for inference
N_PREDICTIONS = int(sys.argv[3]) if len(sys.argv) == 4 else len(tempo_data)
N_PREDICTIONS = len(tempo_data) if N_PREDICTIONS > len(tempo_data)  else N_PREDICTIONS
inputs_targets = [tempo_data[i] for i in range(N_PREDICTIONS)]
inputs = torch.cat([torch.unsqueeze(input = input_target[0], dim = 0) for input_target in inputs_targets], dim = 0) # tempo_nn expects (batch_size, num_channels, frequency, time) [4-dimensions], so we add the batch size dimension here with unsqueeze()
targets = torch.cat([input_target[1] for input_target in inputs_targets], dim = 0).view(N_PREDICTIONS, 1)
del inputs_targets

# make an inference
print("Making predictions...")
print("----------------------------------------------------------------")
tempo_nn.eval()
with torch.no_grad():
    predictions = tempo_nn(inputs).view(N_PREDICTIONS, 1)

# print results
error = torch.abs(input = predictions - targets).numpy()
for i in range(N_PREDICTIONS):
    print(f"Case {i + 1}: Predicted = {predictions[i].item():.2f}, Expected = {targets[i].item():.2f}, % Difference = {error[i].item():.2f}%")
print("----------------------------------------------------------------")
print(f"Average Error: {mean(error):.2f}")
print(f"5% percentile:  {percentile(error, q = 5):.2f}")
print(f"10% percentile: {percentile(error, q = 10):.2f}")
print(f"25% percentile: {percentile(error, q = 25):.2f}")
print(f"50% percentile: {percentile(error, q = 50):.2f}")
print(f"75% percentile: {percentile(error, q = 75):.2f}")
print(f"90% percentile: {percentile(error, q = 90):.2f}")
print(f"95% percentile: {percentile(error, q = 95):.2f}")

# output percentile plot
percentiles = range(0, 101)
percentile_values = [float(percentile(error, q = i)) for i in percentiles]
plt.plot(percentiles, percentile_values, "-b")
plt.xlabel("Percentile")
plt.ylabel("Difference")
plt.title("Validation Data Percentiles")
plt.savefig(join(dirname(NN_FILEPATH), "percentile.png")) # save image

##################################################