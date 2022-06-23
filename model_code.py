# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
import numpy as np
from datetime import datetime
import visualkeras
from keras.callbacks import CSVLogger
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import pandas as pd
import glob
import datetime
import os
import utils
from tqdm.keras import TqdmCallback

dir_with_examples = training_set_folder = "./data/"

# Model configuration
batch_size = 25
img_width, img_height, img_num_channels = 100, 100, 3
loss_function = sparse_categorical_crossentropy
no_classes = 21
no_epochs = 10  # 50
optimizer = Adam()
verbosity = 1
checkpoint_path = "./data/model"

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)


images = []
labels = []

input_shape = (img_width, img_height, img_num_channels)


train_datagen, test_datagen = utils.get_test_data()


def get_model():
    # Create the model
    model = Sequential()
    model.add(
        Conv2D(16, kernel_size=(5, 5), activation="relu", input_shape=input_shape)
    )
    model.add(Conv2D(32, kernel_size=(5, 5), activation="relu"))
    model.add(Conv2D(64, kernel_size=(5, 5), activation="relu"))
    model.add(Conv2D(128, kernel_size=(5, 5), activation="relu"))
    model.add(Flatten())
    model.add(Dense(16, activation="relu"))
    model.add(Dense(no_classes, activation="softmax"))

    # Display a model summary
    model.summary()

    # Compile the model
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

    return model


model = get_model()


# callback to stop training if accuracy reaches 0.95
class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") >= 9.5e-1:
            self.model.stop_training = True


# callback to save mode on every epoch and at end of traiing
class MyModelCheckpoint(Callback):
    def __init__(self, freq=1, directory="./data/model/"):
        super().__init__()
        self.freq = freq
        self.directory = directory

    def on_epoch_begin(self, epoch, logs=None):
        if self.freq > 0 and epoch % self.freq == 0:
            model.save_weights("data/model/lastweights.h5")
            now = datetime.datetime.now()
            weights_path = (
                f"{self.directory}/weights_" + now.strftime("%Y%m%d%H%M") + ".h5"
            )
            model_path = f"{self.directory}/model_" + now.strftime("%Y%m%d%H%M") + ".h5"
            model.save_weights(weights_path)
            model.save(model_path)

    def on_train_end(self, logs=None):
        model.save_weights("data/model/lastweights.h5")
        now = datetime.datetime.now()
        weights_path = f"{self.directory}weights_" + now.strftime("%Y%m%d%H%M") + ".h5"
        model_path = f"{self.directory}/model_" + now.strftime("%Y%m%d%H%M") + ".h5"
        model.save_weights(weights_path)
        model.save(model_path)


# callback to log history to file
csv_logger = CSVLogger("model_history_log.csv", append=True)

keras_callbacks = [
    # stop if accuracy is not increasing
    EarlyStopping(monitor="accuracy", patience=3, mode="max", min_delta=0.01),
    # save models every epoch
    MyModelCheckpoint(directory="./data/model/", freq=1),
    # tqdm progress bar
    TqdmCallback(verbose=2),
    # log history to file
    csv_logger,
    # stop if accuracy reaches 0.9+
    CustomCallback(),
]


def train_model(model):
    # Start training
    history = model.fit(
        train_datagen,
        epochs=no_epochs,
        shuffle=False,
        verbose=0,
        callbacks=keras_callbacks,
    )

    print(history)


print("test acc:", test_acc)  #
