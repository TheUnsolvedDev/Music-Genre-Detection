import os
import librosa
import numpy as np
from keras.models import load_model
import tensorflow as tf

#for gpu utilisation
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

genres = {0: "metal", 1: "disco", 2: "classical", 3: "hiphop", 4: "jazz", 
            5: "country", 6: "pop", 7: "blues", 8: "reggae", 9: "rock"}
song_samples = 660000

def load_song(filepath):
    y, sr = librosa.load(filepath)
    y = y[:song_samples]
    return y, sr

def splitsongs(X, window = 0.05, overlap = 0.5):
    temp_X = []
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        if s.shape[0] != chunk:
            continue

        temp_X.append(s)

    return np.array(temp_X)

def to_melspec(songs, n_fft=1024, hop_length=256):
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft,
        hop_length=hop_length, n_mels=128)[:,:,np.newaxis]
    tsongs = map(melspec, songs)
    return np.array(list(tsongs))

       
def get_genre(path, debug=False):
    model = load_model('detector.h5')
    
    y = load_song(path)[0]
    predictions = []
    spectro = []
    signals = splitsongs(y)
    spec_array = to_melspec(signals)
    spectro.extend(spec_array)
    spectro = np.array(spectro)

    pr = np.array(model.predict(spectro))
    predictions = np.argmax(pr, axis=1)
    if debug:
        # print('Load audio:', path)
        # print("\nFull Predictions:")
        # for p in pr: print(list(p))
        # print("\nPredictions:\n{}".format(predictions))
        # print("Confidences:\n{}".format([round(x, 2) for x in np.amax(pr, axis=1)]))
        print("\nOutput Predictions:\n{}\nPredicted class:".format(np.mean(pr, axis=0)))
    
    return genres[np.bincount(predictions).argmax()] # list(np.mean(pr, axis=0))



def to_melspectrogram(songs, n_fft=1024, hop_length=256):
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft,
        hop_length=hop_length, n_mels=128)[:,:,np.newaxis]

    tsongs = map(melspec, songs)
    return np.array(list(tsongs))

if __name__ == '__main__':
    for i in os.listdir('./testings'):

        print(get_genre('./testings/'+i, True))
        print(i)
