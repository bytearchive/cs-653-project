from keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow import keras
from keras.models import Sequential
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def predict(model: Sequential, validation_data_generator):
    print("Started evaluation")
    scores = model.evaluate(
        validation_data_generator, steps=validation_data_generator.n / 25, verbose=1
    )
    print("Evaluation completed")
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("Started prediction")

    y_pred = model.predict_generator(
        validation_data_generator, steps=validation_data_generator.n / 25, verbose=1
    )
    y_pred = y_pred.argmax(axis=1)
    y_true = validation_data_generator.classes.reshape(validation_data_generator.n, 1)

    accuracy = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred, average="weighted")
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average="weighted")
    print("Metrics calculated!")

    print("accuracy=")
    print(accuracy)
    print("recall=")
    print(recall)
    print("precision=")
    print(precision)
    print("confusion matrix=")
    print(confusion_matrix)


def load_model(path: str, model: Sequential):
    model.load_weights(path)


def plot_visalKeras(model):
    from PIL import ImageFont

    visualkeras.layered_view(model, legend=True, font=font)


def get_test_data(training_set_folder="./data/train/"):

    image_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
    )

    train_datagen = image_generator.flow_from_directory(
        training_set_folder,
        save_format="jpeg",
        batch_size=25,
        target_size=(100, 100),
        class_mode="sparse",
        subset="training",
    )

    test_datagen = image_generator.flow_from_directory(
        training_set_folder,
        target_size=(100, 100),
        batch_size=25,
        subset="validation",
        class_mode="sparse",
    )

    return train_datagen, test_datagen
