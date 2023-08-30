# README
# Phillip Long
# August 1, 2023

# Uses a neural network to make predictions of songs' tempos.

# python ./tempo_inferences.py labels_filepath nn_filepath
# python /dfs7/adl/pnlong/artificial_dj/determine_tempo/tempo_inferences.py "/dfs7/adl/pnlong/artificial_dj/data/tempo_data.cluster.tsv" "/dfs7/adl/pnlong/artificial_dj/data/tempo_nn.classification.pth"


# IMPORTS
##################################################
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from numpy import percentile
import matplotlib.pyplot as plt
from tempo_dataset import tempo_dataset, TEMPO_RANGE, get_tempo, get_tempo_index # import dataset class
from tempo_neural_network import tempo_nn, BATCH_SIZE # import neural network class
# sys.argv = ("./tempo_inferences.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pth")
##################################################


# CONSTANTS
##################################################
LABELS_FILEPATH = sys.argv[1]
NN_FILEPATH = sys.argv[2]
##################################################


# RELOAD MODEL
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

# helper function to convert raw labels (floats in BPM) into class indicies
def labels_to_tempo_indicies(labels):
    return torch.tensor(data = list(map(lambda label: get_tempo_index(tempo = label), labels)), dtype = torch.uint8)

##################################################


# MAKE PREDICTIONS
##################################################

# make an inference
with torch.no_grad():
            
    # set to evaluation mode
    tempo_nn.eval()

    # validation loop
    tempos = range(TEMPO_RANGE[0], TEMPO_RANGE[1] + 1)
    confusion_matrix = torch.zeros(len(tempos), len(tempos), dtype = torch.float32).to(device) # rows = actual, columns = prediction
    error = torch.tensor(data = [], dtype = torch.float32).to(device)
    for inputs, raw_labels in tqdm(data_loader, desc = "Making predictions"):

        # register inputs and labels with device
        raw_labels = raw_labels.view(-1).to(device)
        inputs, labels = inputs.to(device), labels_to_tempo_indicies(labels = raw_labels).view(-1).to(device)

        # forward pass: compute predictions on input data using the model
        predictions = tempo_nn(inputs)
        predictions = torch.argmax(input = predictions, dim = 1, keepdim = True).view(-1) # convert to class indicies

        # add to confusion matrix
        confusion_matrix += torch.tensor(data = [[sum(predicted_tempo_index == predictions[labels == actual_tempo_index]) for predicted_tempo_index in range(len(tempos))] for actual_tempo_index in range(len(tempos))], dtype = confusion_matrix.dtype).to(device)
        
        # add error to running count of all the errors in the validation dataset
        predictions = torch.tensor(data = list(map(lambda i: get_tempo(index = i), predictions)), dtype = torch.float32).view(-1).to(device) # convert predicted indicies into actual predicted tempos
        error_batch = torch.abs(input = predictions - raw_labels)
        error = torch.cat(tensors = (error, error_batch), dim = 0)

print("----------------------------------------------------------------")

# normalize confusion matrix
normalized_confusion_matrix = {
    "precision": confusion_matrix / torch.sum(input = confusion_matrix, axis = 0).view(1, -1),
    "recall": confusion_matrix / torch.sum(input = confusion_matrix, axis = 1).view(-1, 1)
}

##################################################


# PRINT RESULTS
##################################################

print(f"Average Error: {torch.mean(input = error).item():.2f}")

# calculate percentiles
percentiles = range(0, 101)
percentile_values = percentile(error.numpy(force = True), q = percentiles)
print(f"Minimum Error: {percentile_values[0]:.2f}")
print(*(f"{i}% Percentile: {percentile_values[i]:.2f}" for i in (5, 10, 25, 50, 75, 90, 95)), sep = "\n")
print(f"Maximum Error: {percentile_values[100]:.2f}")
print("----------------------------------------------------------------")

##################################################


# MAKE PLOTS
##################################################

fig, axes = plt.subplot_mosaic(mosaic = [["confusion", "percentiles"], ["normalized", "percentiles"]], constrained_layout = True, figsize = (12, 8))
fig.suptitle("Testing the Tempo Neural Network")

##################################################


# CONFUSION MATRICES
##################################################

tick_labels_step = 10

# plot confusion matrix
confusion_plot_temp = axes["confusion"].imshow(confusion_matrix, aspect = "auto", origin = "upper", cmap = "Blues")
fig.colorbar(confusion_plot_temp, ax = axes["confusion"], label = "n", location = "right")
axes["normalized"].set_xticks(ticks = range(0, len(tempos), tick_labels_step), labels = range(tempos.start, tempos.stop, tick_labels_step))
axes["confusion"].set_ylabel("Actual Tempos")
axes["confusion"].set_yticks(ticks = range(0, len(tempos), tick_labels_step), labels = range(tempos.start, tempos.stop, tick_labels_step))
axes["confusion"].set_title("Confusion Matrix")

# plot normalized confusion matrix (either precision or recall)
normalized_confusion_matrix_type = "precision"
normalized_confusion_plot_temp = axes["normalized"].imshow(normalized_confusion_matrix[normalized_confusion_matrix_type], aspect = "auto", origin = "upper", cmap = "Reds")
fig.colorbar(normalized_confusion_plot_temp, ax = axes["normalized"], label = "", location = "right")
axes["normalized"].set_xticks(ticks = range(0, len(tempos), tick_labels_step), labels = range(tempos.start, tempos.stop, tick_labels_step))
axes["normalized"].set_xlabel("Predicted Tempos")
axes["confusion"].sharex(other = axes["normalized"]) # share the x-axis labels
axes["normalized"].set_ylabel("Actual Tempos")
axes["normalized"].set_yticks(ticks = range(0, len(tempos), tick_labels_step), labels = range(tempos.start, tempos.stop, tick_labels_step))
axes["normalized"].set_title(normalized_confusion_matrix_type.title())

##################################################


# PERCENTILE PLOT
##################################################

axes["percentiles"].plot(percentiles, percentile_values, color = "tab:blue", linestyle = "-")
axes["percentiles"].set_xlabel("Percentile")
axes["percentiles"].set_ylabel("Error")
axes["percentiles"].set_title("Test Data Percentiles")
axes["percentiles"].grid()

##################################################


# OUTPUT
##################################################

print("Outputting plot...")
fig.savefig(".".join(NN_FILEPATH.split(".")[:-1]) + ".test.png", dpi = 240) # save image

##################################################
