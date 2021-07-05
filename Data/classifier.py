from tensorflow.keras.utils import Sequence
import tensorflow as tf
from copy import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_extraction import data
import datetime
import keras


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class GTZANGenerator(Sequence):
    def __init__(self, X, y, batch_size=64, is_test=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.is_test = is_test

    def __len__(self):
        return int(np.ceil(len(self.X)/self.batch_size))

    def __getitem__(self, index):
        signals = self.X[index*self.batch_size:(index+1)*self.batch_size]
        if not self.is_test:
            signals = self.__augment(signals)
        return signals, self.y[index*self.batch_size:(index+1)*self.batch_size]

    def __augment(self, signals, hor_flip=0.5, random_cutout=0.5):
        spectrograms = []
        for s in signals:
            signal = copy(s)
            if np.random.rand() < hor_flip:
                signal = np.flip(signal, 1)

            if np.random.rand() < random_cutout:
                lines = np.random.randint(signal.shape[0], size=3)
                cols = np.random.randint(signal.shape[0], size=4)
                signal[lines, :, :] = -80
                signal[:, cols, :] = -80

            spectrograms.append(signal)
        return np.array(spectrograms)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        np.random.shuffle(self.indexes)
        return None


class classifier:
    def __init__(self, train_data, train_label, test_data, test_label):
        mean = np.mean(train_data, axis=(0, 1, 2))
        stddev = np.std(train_data, axis=(0, 1, 2))

        self.train_data = (train_data-mean)/stddev
        print(type(self.train_data))
        self.train_label = tf.one_hot(train_label, depth=10)
        self.test_data = (test_data-mean)/stddev
        self.test_label = tf.one_hot(test_label, depth=10)

        batch_size = 128
        self.train_generator = GTZANGenerator(
            self.train_data, self.train_label)
        self.steps_per_epoch = np.ceil(len(self.train_data)/batch_size)

        self.validation_generator = GTZANGenerator(
            self.test_data, self.test_label)
        self.val_steps = np.ceil(len(self.test_data)/batch_size)

    # @tf.function
    def model_for_numbers(self):
        input = tf.keras.Input(shape=self.train_data.shape[1])

        layer1 = tf.keras.layers.Dense(
            1024, activation='relu', kernel_initializer='he_uniform')(input)
        layer1 = tf.keras.layers.Dropout(0.4)(layer1)
        layer1 = tf.keras.layers.BatchNormalization()(layer1)

        layer2 = tf.keras.layers.Dense(
            512, activation='relu', kernel_initializer='he_uniform')(layer1)
        layer2 = tf.keras.layers.Dropout(0.4)(layer2)
        layer2 = tf.keras.layers.BatchNormalization()(layer2)

        layer3 = tf.keras.layers.Dense(
            256, activation='relu', kernel_initializer='he_uniform')(layer2)
        layer3 = tf.keras.layers.Dropout(0.4)(layer3)
        layer3 = tf.keras.layers.BatchNormalization()(layer3)

        layer4 = tf.keras.layers.Dense(
            128, activation='relu', kernel_initializer='he_uniform')(layer3)
        layer4 = tf.keras.layers.Dropout(0.4)(layer4)
        layer4 = tf.keras.layers.BatchNormalization()(layer4)

        layer5 = tf.keras.layers.Dense(
            64, activation='relu', kernel_initializer='he_uniform')(layer4)
        layer5 = tf.keras.layers.Dropout(0.4)(layer5)
        layer5 = tf.keras.layers.BatchNormalization()(layer5)

        output_layer = tf.keras.layers.Dense(10, activation='softmax')(layer5)

        model = tf.keras.Model(inputs=input, outputs=output_layer)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
        return model

    # @tf.function
    def model_for_image(self):
        input_shape = self.train_data[0].shape

        input = tf.keras.Input(input_shape)
        x = tf.keras.layers.Conv2D(16, (3, 3), strides=(
            1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(input)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(32, (3, 3), strides=(
            1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(
            1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(
            1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(256, (3, 3), strides=(
            1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            4096, activation='relu', kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(4096, activation='relu',
                               kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        output_layer = tf.keras.layers.Dense(10,
                                             activation='softmax',
                                             kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)

        model = tf.keras.Model(inputs=input, outputs=output_layer)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])

        return model

    def train(self):
        model = self.model_for_image()
        model.summary()
        reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.95,
            patience=3,
            verbose=1,
            mode='min',
            min_delta=0.0001,
            cooldown=2,
            min_lr=1e-5
        )
        history = model.fit(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            batch_size=50,
            epochs=500,
            callbacks=[tensorboard_callback, reduceLROnPlat],
            validation_data=self.validation_generator,
            validation_steps=self.val_steps,
            verbose=1
        )
        model.save('detector.h5')
        fig, axs = plt.subplots(2)

        # create accuracy sublpot
        axs[0].plot(history.history["accuracy"], label="train accuracy")
        axs[0].plot(history.history["val_accuracy"], label="test accuracy")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend(loc="lower right")
        # axs[0].set_title("Accuracy eval")

        # create error sublpot
        axs[1].plot(history.history["loss"], label="train error")
        axs[1].plot(history.history["val_loss"], label="test error")
        axs[1].set_ylabel("Error")
        axs[1].set_xlabel("Epoch")
        axs[1].legend(loc="upper right")
        # axs[1].set_title("Error eval")

        plt.show()


if __name__ == '__main__':
    sample = data(path='genres_original')
    (train_data, train_label), (test_data, test_label) = sample.load_image_data()
    # NN object for classification
    NN = classifier(train_data, train_label, test_data, test_label)
    NN.train()
