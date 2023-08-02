# README
# Phillip Long
# August 1, 2023

# Creates and trains a linear regression neural network in PyTorch.
# Given an audio file as input, it outputs a single number representing the song's tempo in Beats per Minute (BPM).

# python ./tempo_neural_network.py


# IMPORTS
##################################################
import torch
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
import torchaudio
from determine_tempo.tempo_dataset import tempo_dataset # dataset class
##################################################


# NEURAL NETWORK CLASS
##################################################
class CNNNetwork(nn.Module):

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

if __name__ == "__main__":

    # constants
    BATCH_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 0.001
    ANNOTATIONS_FILE = "/Volumes/Seagate/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/Volumes/Seagate/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    N_SAMPLES = 22050
    

    # determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    # summarize 
    cnn = CNNNetwork().to(device)
    summary(model = cnn, input_size = (1, 64, 44)) # input_size = (# of channels, # of mels [frequency axis], time axis)



def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size = batch_size)
    return train_dataloader


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimizer.zero_grad() # zero the gradients
        loss.backward() # conduct backpropagation
        optimizer.step() # update parameters
    
    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("********************")
    
    print("Training is done.")


if __name__ == "__main__":
    print("Starting program.")    

    # determine what device we are using
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    
    # instantiate our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 1024 // 2, n_mels = 64) # instantiate melspectrogram transformation
    usd = UrbanSoundDataset(annotations_file = ANNOTATIONS_FILE,
                            audio_dir = AUDIO_DIR,
                            target_sample_rate = SAMPLE_RATE,
                            n_samples = N_SAMPLES,
                            device = device,
                            transformation = mel_spectrogram)
    train_dataloader = create_data_loader(train_data = usd, batch_size = BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr = LEARNING_RATE)

    # train
    print("********************")
    train(model = cnn, data_loader = train_dataloader, loss_fn = loss_fn, optimizer = optimizer, device = device, epochs = EPOCHS)

    # store trained model
    output_filepath = "/Users/philliplong/Desktop/Coding/mcauley_lab_prep/audio_processing_pytorch/data/cnn.pth"
    torch.save(cnn.state_dict(), output_filepath)
    print(f"Model trained and stored at {output_filepath}.")
