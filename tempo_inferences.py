# README
# Phillip Long
# August 1, 2023

# Uses a neural network to make predictions of songs' tempos.

# python ./tempo_inferences.py labels_filepath nn_filepath


# IMPORTS
##################################################
import sys
import torch
import torchaudio
from tempo_dataset import tempo_dataset, SAMPLE_RATE, SAMPLE_DURATION 
from tempo_neural_network import tempo_nn # import convolution neural network Network class
# sys.argv = ("./tempo_inferences.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pth")
##################################################


# CONSTANTS
##################################################
N_PREDICTIONS = 20
LABELS_FILEPATH = sys.argv[1]
NN_FILEPATH = sys.argv[2]
##################################################


# RELOAD MODEL AND MAKE PREDICTIONS
##################################################
# load back the model
tempo_nn = tempo_nn()
state_dict = torch.load(NN_FILEPATH)
tempo_nn.load_state_dict(state_dict)
print("Imported neural network parameters.")

# instantiate our dataset object
tempo_data = tempo_dataset(labels_filepath = LABELS_FILEPATH,
                           set_type = "validate",
                           target_sample_rate = SAMPLE_RATE,
                           sample_duration = SAMPLE_DURATION,
                           device = "cpu",
                           transformation = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 1024 // 2, n_mels = 64)
                           )

# get a sample from the urban sound dataset for inference
N_PREDICTIONS = len(tempo_data) if len(tempo_data) < N_PREDICTIONS else N_PREDICTIONS
inputs_targets = [tempo_data[i] for i in range(N_PREDICTIONS)]
inputs = torch.cat([torch.unsqueeze(input = input_target[0], dim = 0) for input_target in inputs_targets], dim = 0) # tempo_nn expects (batch_size, num_channels, frequency, time) [4-dimensions], so we add the batch size dimension here with unsqueeze()
targets = [input_target[1] for input_target in inputs_targets]
del inputs_targets

# make an inference
print("Making predictions.")
print("********************")
tempo_nn.eval()
with torch.no_grad():
    predicted = tempo_nn(inputs.to(torch.float32))
for i in range(N_PREDICTIONS):
    print(f"Case {i + 1}: Predicted = '{predicted[i]}', Expected = '{targets[i]}', Correct = {predicted[i] == targets[i]}")
print("********************")
accuracy = sum(predicted[i] == targets[i] for i in range(N_PREDICTIONS)) / N_PREDICTIONS
print(f"Accuracy: {(100 * accuracy):.2f}%")
##################################################