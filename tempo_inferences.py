# README
# Phillip Long
# August 1, 2023

# Uses a neural network to make predictions of songs' tempos.

# python ./tempo_inferences.py labels_filepath nn_filepath
# python /dfs7/adl/pnlong/artificial_dj/determine_tempo/tempo_inferences.py "/dfs7/adl/pnlong/artificial_dj/data/tempo_data.cluster.tsv" "/dfs7/adl/pnlong/artificial_dj/data/tempo_nn.pth"


# IMPORTS
##################################################
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from numpy import percentile
import matplotlib.pyplot as plt
from tempo_dataset import tempo_dataset, get_tempo # import dataset class
from tempo_neural_network import tempo_nn, BATCH_SIZE # import neural network class
# sys.argv = ("./tempo_inferences.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pth")
##################################################


# CONSTANTS
##################################################
LABELS_FILEPATH = sys.argv[1]
NN_FILEPATH = sys.argv[2]
##################################################


# RELOAD MODEL AND MAKE PREDICTIONS
##################################################

# I want to do all the predictions on CPU, GPU seems a bit much
print("----------------------------------------------------------------")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

# load back the model
tempo_nn = tempo_nn(nn_filepath = NN_FILEPATH, device = device).to(device)
print("Imported neural network parameters.")

# instantiate our dataset object and data loader
data = tempo_dataset(labels_filepath = LABELS_FILEPATH, set_type = "test", device = device)
data_loader = DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle = True)

# make an inference
with torch.no_grad():
            
    # set to evaluation mode
    tempo_nn.eval()

    # validation loop
    error = torch.tensor(data = [], dtype = torch.float32).to(device)
    for inputs, labels in tqdm(data_loader, desc = "Making predictions"):

        # register inputs and labels with device
        inputs, labels = inputs.to(device), labels.to(device)

        # forward pass: compute predictions on input data using the model
        predictions = tempo_nn(inputs)
        predictions = torch.argmax(input = predictions, dim = 1, keepdim = True).view(-1) # convert to class indicies
    
        # add error to running count of all the errors in the validation dataset
        predictions = torch.tensor(data = list(map(lambda i: get_tempo(index = i), predictions)), dtype = torch.float32).to(device) # convert predicted indicies into actual predicted tempos
        error_batch = torch.abs(input = predictions.view(-1) - labels.view(-1))
        error = torch.cat(tensors = (error, error_batch), dim = 0)

print("----------------------------------------------------------------")

# print results
print(f"Average Error: {torch.mean(input = error).item():.2f}")

# calculate percentiles
percentiles = range(0, 101)
percentile_values = percentile(error.numpy(force = True), q = percentiles)
print(f"Minimum Error: {percentile_values[0]:.2f}")
print(*(f"{i}% Percentile: {percentile_values[i]:.2f}" for i in (5, 10, 25, 50, 75, 90, 95)), sep = "\n")
print(f"Maximum Error: {percentile_values[100]:.2f}")
print("----------------------------------------------------------------")

# output percentile plot
plt.plot(percentiles, percentile_values, color = "b", linestyle = "-")
plt.xlabel("Percentile")
plt.ylabel("Error")
plt.title("Test Data Percentiles")
plt.savefig(".".join(NN_FILEPATH.split(".")[:-1]) + ".test.png") # save image
print("Outputting percentile plot...")

##################################################
