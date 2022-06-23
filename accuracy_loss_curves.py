# coding: utf-8
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_train_acc(log_file="model_history_log.csv"):
    history = pd.read_csv(log_file)
    plt.plot(history["accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train"], loc="upper left")
    plt.savefig("./plots/model_accuracy.png")


def plot_train_loss(log_file="model_history_log.csv"):
    history = pd.read_csv(log_file)
    plt.plot(history["loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train"], loc="upper left")
    plt.savefig("./plots/model_loss.png")


if __name__ == "__main__":
    plot_train_acc()
    plot_train_loss()
