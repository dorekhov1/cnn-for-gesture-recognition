import matplotlib.pyplot as plt
import pandas as pd
import torch
from util import *


def load_csv(type, model_path, model_name):

    train_file = '{}/train_{}_{}.csv'.format(model_path, type, model_name)
    val_file = '{}/val_{}_{}.csv'.format(model_path, type, model_name)

    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)

    return train_data, val_data


def plot_graph(type, model_path, model_name, train_data, val_data):
    """
    Plot the training loss/error curve given the data from CSV
    """
    plt.figure()
    type_title = "Error" if type == "err" else "Loss"
    plt.title("{} over training epochs\n{}".format(type_title, model_name))
    plt.plot(train_data["epoch"], train_data["train_{}".format(type)], label="Train")
    plt.plot(val_data["epoch"], val_data["val_{}".format(type)], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(type_title)
    plt.legend(loc='best')
    plt.savefig("{}/{}_{}.png".format(model_path, type, model_name))

    return


def plot_train(model_path, model_name):

    train_err_data, val_err_data = load_csv('err', model_path, model_name)
    train_loss_data, val_loss_data = load_csv('loss', model_path, model_name)

    print("Final training error: {0:.3f}% | Final validation error: {1:.3f}%"
          .format(train_err_data["train_err"].iloc[-1]*100, val_err_data["val_err"].iloc[-1]*100))
    print("Final training loss: {0:.5f} | Final validation loss: {1:.5f}".format(train_loss_data["train_loss"].iloc[-1],
          val_loss_data["val_loss"].iloc[-1]))

    plot_graph("err", model_path, model_name, train_err_data, val_err_data)
    plot_graph("loss", model_path, model_name, train_loss_data, val_loss_data)
