# determine_tempo
Determine the tempo (in BPM) of a given audio sample.

---

## Background

DJs need to know the tempo of a song, typically measured in Beats Per Minute (BPM), if they want to incorporate the track into their mix. For instance, a fast song will not mix well with a slow song, but two fast songs can sound quite good together. I will use machine learning to take an audio file (.mp3) as input and output the song's BPM.


## Data

I will use my personal song library (>2000 songs) to train this neural network, excluding classical and other genres of music that tend to have variable tempos. I will create a data table that contains information on each song's BPM, as well as their key for later use. This will probably involve creating a webscraper that takes an .MP3 as input and uses the MP3's metadata to search a website like [TuneBat](https://tunebat.com/) for important song information. I will divide my data 70%-20%-10%: 70% training data, 20% cross validation data, and the remaining 10% for measuring actual performance metrics. I am still deciding whether I will use raw-waveform or melspectrogram data for training my neural network.


## Machine Learning

I will use PyTorch to create a Convolutional Neural Network (CNN) that takes an .MP3 file as input and outputs a single value, the song's tempo measured in BPM. This is a linear regression problem. The hope is that a CNN can better detect patterns in the audio data such as a kick drum or bassline that are useful in maintaining (and thus determining) a song's tempo.

I can use two approaches for this neural network:

- **Window Method**: Because the input of a CNN needs to be of a fixed size, I can train my network on 8-second (or any other amount of time) snippets of audio: *windows*. Using the same window size on which my neural network was trained, I would then scan over a song and for each window output a predicted tempo. The average song would contain 60 to 180 windows (and thus predicted BPM values), so I would ultimately take the median or mode tempo from this list of BPMs for a final value. The benefit of this method is that there is potential for a lot of training data, as the average song can be divided up into, as mentioned previously, 60 to 180 data points.
    - An option for speeding up this method is that once the neural network is trained, instead of sliding a fixed-size window over the *entire* 60- to 180-window song, I can randomly select `n` windows (around ~10) from the song, run each of these windows through the neural network, and then take the median/mode BPM of this subset of values.
- **LSTM Method**. The downside to traditional neural networks is that the input is a fixed size; this is not good when dealing with music, since songs vary significantly in length. To deal with inputs of variable size, Recurrent Neural Networks (RNNs) are used. Vanilla RNNs are quite limited on their own, so an architecture called Long Short Term Memory (LSTM) is used as an improvement. An even greater improvement is to utilize Attention, though this could be quite difficult to implement given my level. By using LSTM at the least, my neural network can take an entire song as input without the need for windowing and output a single value, the song's tempo.

---

## Software

### *.py*