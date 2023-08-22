# README
# Phillip Long
# August 21, 2023

# Makes plots describing the training of the neural network.

# python ./training_plots.py history_filepath percentiles_history_filepath output_filepath
# python /dfs7/adl/pnlong/artificial_dj/determine_tempo/training_plots.py "/dfs7/adl/pnlong/artificial_dj/data/tempo_nn.pretrained.history.tsv" "/dfs7/adl/pnlong/artificial_dj/data/tempo_nn.pretrained.percentiles_history.tsv" "/dfs7/adl/pnlong/artificial_dj/data/tempo_nn.pretrained.png"


# IMPORTS
##################################################
import sys
import pandas as pd
import matplotlib.pyplot as plt
# sys.argv = ("./training_plots.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pretrained.history.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pretrained.percentiles_history.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pretrained.png")
##################################################


# IMPORT FILEPATHS, LOAD FILES
##################################################

# make arguments
OUTPUT_FILEPATH_HISTORY = sys.argv[1]
OUTPUT_FILEPATH_PERCENTILES_HISTORY = sys.argv[2]
OUTPUT_FILEPATH = sys.argv[3]

# load in tsv files that have been generated
history = pd.read_csv(OUTPUT_FILEPATH_HISTORY, sep = "\t", header = 0, index_col = False)
percentiles_history = pd.read_csv(OUTPUT_FILEPATH_PERCENTILES_HISTORY, sep = "\t", header = 0, index_col = False)

##################################################


# CREATE PLOT
##################################################

# plot loss and percentiles per epoch
fig, axes = plt.subplot_mosaic([["loss", "percentiles_history"], ["accuracy", "percentiles_history"]], constrained_layout = True, figsize = (12, 8))
fig.suptitle("Tempo Neural Network")
colors = ["b", "r", "g", "c", "m", "y", "k"]

##################################################


# PLOT LOSS
##################################################

for set_type, color in zip(("train_loss", "validate_loss"), colors[:2]):
    axes["loss"].plot(history["epoch"], history[set_type], color = color, linestyle = "solid", label = set_type.split("_")[0].title())
axes["loss"].set_xlabel("Epoch")
axes["loss"].set_ylabel("Loss")
axes["loss"].legend(loc = "upper right")
axes["accuracy"].set_title("Learning Curve")

##################################################


# PLOT ACCURACY
##################################################

for set_type, color in zip(("train_accuracy", "validate_accuracy"), colors[:2]):
    axes["accuracy"].plot(history["epoch"], history[set_type], color = color, linestyle = "dashed", label = set_type.split("_")[0].title())
axes["loss"].sharex(axes["accuracy"])
axes["accuracy"].set_ylabel("Average Error")
axes["accuracy"].legend(loc = "upper right")
axes["accuracy"].set_title("Average Error")

##################################################


# PLOT PERCENTILES PER EPOCH
##################################################

# plot percentiles per epoch (final 5 epochs)
epochs = sorted(pd.unique(percentiles_history["epoch"]))
n_epochs = min(5, len(epochs), len(colors))
colors = colors[:n_epochs]
percentiles_history = percentiles_history[percentiles_history["epoch"] > (max(percentiles_history["epoch"] - n_epochs))]
for i, epoch in enumerate(epochs):
    percentile_at_epoch = percentiles_history[percentiles_history["epoch"] == epoch]
    axes["percentiles_history"].plot(percentile_at_epoch["percentile"], percentile_at_epoch["value"], color = colors[i], linestyle = "solid", label = epoch)
axes["percentiles_history"].set_xlabel("Percentile")
axes["percentiles_history"].set_ylabel("Error")
axes["percentiles_history"].legend(title = "Epoch", loc = "upper left")
axes["percentiles_history"].grid()
axes["percentiles_history"].set_title("Validation Data Percentiles")

##################################################


# SAVE
##################################################

# save figure
fig.savefig(OUTPUT_FILEPATH, dpi = 180) # save image

##################################################
