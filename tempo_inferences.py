# README
# Phillip Long
# August 1, 2023

# Uses a neural network to make predictions of songs' tempos.

# python ./tempo_inferences.py


import torch
import torchaudio
from urbansounddataset import UrbanSoundDataset # import dataset class
from cnn import CNNNetwork # import convolution neural network Network class
from cnn_train import ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, N_SAMPLES


N_PREDICTIONS = 20


class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, inputs, targets, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(inputs.to(torch.float32))
        predicted_index = predictions.argmax(dim = 1)
        predicted = [class_mapping[i] for i in predicted_index]
        expected = [class_mapping[i] for i in targets]
    return predicted, expected



if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("/Users/philliplong/Desktop/Coding/mcauley_lab_prep/audio_processing_pytorch/data/cnn.pth")
    cnn.load_state_dict(state_dict)
    print("Imported neural network parameters.")

    # load urban sound dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 1024 // 2, n_mels = 64) # instantiate melspectrogram transformation
    usd = UrbanSoundDataset(annotations_file = ANNOTATIONS_FILE,
                            audio_dir = AUDIO_DIR,
                            target_sample_rate = SAMPLE_RATE,
                            n_samples = N_SAMPLES,
                            device = "cpu",
                            transformation = mel_spectrogram)

    # get a sample from the urban sound dataset for inference
    sampling_indicies, _ = torch.sort(input = torch.ones(len(usd)).multinomial(N_PREDICTIONS, replacement = False)) # indicies to sample from data
    inputs_targets = [usd[int(i)] for i in sampling_indicies]
    inputs = torch.cat([torch.unsqueeze(input = input_target[0], dim = 0) for input_target in inputs_targets], dim = 0) # CNNNetwork expects (batch_size, num_channels, frequency, time) [4-dimensions], so we add the batch size dimension here with unsqueeze_()
    targets = [input_target[1] for input_target in inputs_targets]
    del inputs_targets

    # make an inference
    print("Making predictions.")
    print("********************")
    predicted, expected = predict(cnn, inputs, targets, class_mapping)
    for i in range(N_PREDICTIONS):
        print(f"Case {sampling_indicies[i] + 1}: Predicted = '{predicted[i]}', Expected = '{expected[i]}', Correct = {predicted[i] == expected[i]}")
    print("********************")
    accuracy = sum(predicted[i] == expected[i] for i in range(N_PREDICTIONS)) / N_PREDICTIONS
    print(f"Accuracy: {(100 * accuracy):.2f}%")