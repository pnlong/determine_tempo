# README
# Phillip Long
# August 1, 2023

# Creates and trains a linear regression neural network in PyTorch.
# Given an audio file as input, it outputs a single number representing the song's tempo in Beats per Minute (BPM).

# python ./tempo_neural_network.py labels_filepath output_filepath


# IMPORTS
##################################################
import sys
from tqdm import tqdm
import torch
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
import torchaudio
from tempo_dataset import tempo_dataset, SAMPLE_RATE, SAMPLE_DURATION # dataset class + some constants
# sys.argv = ("./tempo_neural_network.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pth")
##################################################


# CONSTANTS
##################################################
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
##################################################


# NEURAL NETWORK CLASS
##################################################
class tempo_nn(nn.Module):

    def __init__(self):
        super().__init__()
        # convolutional block 1 -> convolutional block 2 -> convolutional block 3 -> convolutional block 4 -> flatten -> linear -> softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.flatten = nn.Flatten(start_dim = 1)
        self.linear = nn.Linear(in_features = 128 * 5 * 4, out_features = 10) # out_features = # of classes in UrbanSound8K dataset
        # self.softmax = nn.Softmax(dim = 1)
        # Just a note (per PyTorch docs)... when using nn.CrossEntropyLoss() as the loss_fn, it is important to keep the model output as raw logits
        # (ie. do not include the softmax() in the model as the final output layer).
        # I read in a discussion that this is important to reduce the potential for numerical instabilities due to some log-sum-exp equation that is performed.
        # This might be new to the current version of PyTorch (2.0.1).

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        # predictions = self.softmax(logits)
        return logits

##################################################


# MODEL TRAINING FUNCTION
##################################################
# train the whole model
def train(model, data_loader, loss_function, optimizer, device, epochs):

    for _ in tqdm(range(epochs), desc = "Training Neural Network", mininterval = 5): # epoch for loop
        # train an epoch
        for inputs, labels in data_loader:
            # register inputs and labels with device
            inputs, labels = inputs.to(device), labels.to(device)

            # calculate loss
            predictions = model(inputs)
            loss = loss_function(predictions, labels)

            # backpropagate loss and update weights
            optimizer.zero_grad() # zero the gradients
            loss.backward() # conduct backpropagation
            optimizer.step() # update parameters

##################################################


if __name__ == "__main__":

    # CONSTANTS
    ##################################################
    LABELS_FILEPATH = sys.argv[1]
    OUTPUT_FILEPATH = sys.argv[2]
    ##################################################

    # TRAIN NEURAL NETWORK
    ##################################################

    # determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    
    # instantiate our dataset object
    tempo_data = tempo_dataset(labels_filepath = LABELS_FILEPATH,
                               target_sample_rate = SAMPLE_RATE,
                               sample_duration = SAMPLE_DURATION,
                               device = device,
                               transformation = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 1024 // 2, n_mels = 64)
                               )

    # construct model and assign it to device, also summarize 
    print("********************")
    print("Summary of Neural Network:")
    tempo_nn = tempo_nn().to(device)
    summary(model = tempo_nn, input_size = tempo_data[0][0].shape) # input_size = (# of channels, # of mels [frequency axis], time axis)
    print("********************")

    # instantiate data loader, loss function, and optimizer
    data_loader = DataLoader(tempo_data, batch_size = BATCH_SIZE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tempo_nn.parameters(), lr = LEARNING_RATE)

    # train
    train(model = tempo_nn, data_loader = data_loader, loss_function = loss_function, optimizer = optimizer, device = device, epochs = EPOCHS)
    print("Training is done.")

    # store trained model
    torch.save(tempo_nn.state_dict(), OUTPUT_FILEPATH)
    print(f"Model trained and stored at {OUTPUT_FILEPATH}.")
    ##################################################