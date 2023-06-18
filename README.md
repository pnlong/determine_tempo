# determine_tempo
Determine the tempo (in BPM) of a given audio sample.

---

## Background

DJs need to know the tempo of a song, typically measured in Beats Per Minute (BPM), if they want to incorporate the track into their mix. For instance, a fast song will not mix well with a slow song, but two fast songs can sound quite good together. I will use machine learning to take an audio file (.mp3) as input and output the song's BPM.


## Data

I will use my personal song library (>2000 songs) to train this neural network, excluding classical and other genres of music that tend to have variable tempos. I will create a data table that contains information on each song's BPM, as well as their key for later use. This will probably involve creating a webscraper that takes an .MP3 as input and uses the MP3's metadata to search a website like [TuneBat](https://tunebat.com/) for important song information. I will divide my data 70%-20%-10%: 70% training data, 20% cross validation data, and the remaining 10% for measuring actual performance metrics. I am still deciding whether I will use raw-waveform or melspectrogram data for training my neural network.


## Machine Learning

I will use PyTorch to create a convolutional neural network that takes an .MP3 file as input and outputs a single value, the song's tempo measured in BPM. This is a linear regression problem. The hope is that a convolutional network can better detect patterns in the audio data such as a kick drum or bassline that are useful in maintaining (and thus determining) a song's tempo.


---

## Software

### *.py*