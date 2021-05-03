import os
import json
import numpy as np
import pandas as pd
import math
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class data:
    def __init__(self, csv=None, path=None):
        if csv:
            data = pd.read_csv(csv)
            num_classes = len(data['label'].unique())

            label_encoder = LabelEncoder()
            labels = data['label'].unique()
            label_encoder.fit(labels)

            data.insert(60, 'label_id', 9999)
            data.insert(1, 'filename_full', '')
            for i in range(len(data)):
                label = data.loc[i, 'label']
                label_id = label_encoder.transform([label])
                data.loc[i, 'label_id'] = label_id.item()
                data.loc[i, 'filename_full'] = str(data.loc[i, 'filename']).split(
                    '.')[0]+"."+str(data.loc[i, 'filename']).split('.')[1]+"."+str(data.loc[i, 'filename']).split('.')[3]
            data['label_id'] = data['label_id'].astype(int)

            self.features_full = data.drop(
                ['filename', 'filename_full', 'length', 'label', 'label_id'], axis=1)
            self.target_full = data['label_id'].astype('int')

        elif path == 'images_original':
            data = ['blues', 'classical', 'country', 'disco',
                    'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            count = 0
            dirs = [path + '/' + i for i in data]

            self.data, self.labels = [], []
            for d in dirs:
                for file in os.listdir(d):
                    image = cv2.imread(d+'/'+file, 0)
                    image = np.array(image, dtype=np.float32)
                    image = cv2.resize(image, (120, 120)).reshape(120, 120, 1)
                    self.data.append(image)
                    self.labels.append(count)
                count += 1

        elif path == 'genres_original':
            self.gtzan_dir = path
            self.song_samples = 660000
            self.genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
                           'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
            

    def load_image(self):
        features, features_test, target, target_test = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=42, shuffle=True)
        return (np.array(features), np.array(target)), (np.array(features_test), np.array(target_test))

    def load_numerical_data(self):
        features, features_test, target, target_test = train_test_split(
            self.features_full, self.target_full, test_size=0.25, random_state=42, shuffle=True)
        scaler = StandardScaler()
        features = pd.DataFrame(scaler.fit_transform(
            features), columns=features.columns)
        features_test = pd.DataFrame(scaler.transform(
            features_test), columns=features_test.columns)
        return (features, target), (features_test, target_test)

    def load_image_data(self):
        (X_train, y_train), (X_test, y_test) = self.read_data()
        return (X_train, y_train), (X_test, y_test)

    def split_convert(self, X, y):
        arr_specs, arr_genres = [], []

        for fn, genre in zip(X, y):
            signal, sr = librosa.load(fn)
            signal = signal[:self.song_samples]
            signals, y = self.splitsongs(signal, genre)
            specs = self.to_melspectrogram(signals)

            arr_genres.extend(y)
            arr_specs.extend(specs)

        return np.array(arr_specs), np.array(arr_genres)

    def read_data(self):
        src_dir = self.gtzan_dir 
        genres = self.genres
        song_samples = self.song_samples
        arr_fn = []
        arr_genres = []

        for x, _ in genres.items():
            folder = src_dir + "/" + x
            for root, subdirs, files in os.walk(folder):
                for file in files:
                    file_name = folder + "/" + file
                    if file_name == 'genres_original/jazz/jazz.00053.wav':
                        continue
                    # print(arr_fn)
                    arr_fn.append(file_name)
                    arr_genres.append(genres[x])
        X_train, X_test, y_train, y_test = train_test_split(
            arr_fn, arr_genres, test_size=0.15, random_state=42, stratify=arr_genres
        )
        X_train, y_train = self.split_convert(X_train, y_train)
        X_test, y_test = self.split_convert(X_test, y_test)

        return (X_train, y_train), (X_test,  y_test)

    def splitsongs(self, X, y, window=0.05, overlap=0.5):
        temp_X = []
        temp_y = []

        xshape = X.shape[0]
        chunk = int(xshape*window)
        offset = int(chunk*(1.-overlap))

        spsong = [X[i:i+chunk]
                  for i in range(0, xshape - chunk + offset, offset)]
        for s in spsong:
            temp_X.append(s)
            temp_y.append(y)

        return np.array(temp_X), np.array(temp_y)

    def to_melspectrogram(self, songs, n_fft=1024, hop_length=256):
        def melspec(x): return librosa.feature.melspectrogram(x, n_fft=n_fft,
                                                              hop_length=hop_length)[:, :, np.newaxis]

        tsongs = map(melspec, songs)
        return np.array(list(tsongs))


if __name__ == '__main__':
    # datum = data(path='images_original')
    datum = data(path='genres_original')
    (a, b), (c, d) = datum.load_image_data()
    print(a.shape, b.shape, c.shape, d.shape)
    print(type(a))
    plt.imshow(a[0], cmap='hot')
    plt.show()
