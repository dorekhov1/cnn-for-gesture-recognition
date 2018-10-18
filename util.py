import numpy as np
import json

from dataset import GestureDataset
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torch.optim as optim

import torch as torch

import os
from sklearn.model_selection import train_test_split


def load_config(path):

    with open(path) as file:
        config = json.load(file)

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    loss = config["loss"]
    optim = config["optim"]
    model = config["model"]
    split = config["split"]
    s = config["seed"]

    return config, learning_rate, batch_size, num_epochs, loss, optim, model, split, s


def get_model_name(config):

    path = "mdl{}_".format(config["model"])
    path += "ep{}_".format(config["num_epochs"])
    path += "bs{}_".format(config["batch_size"])
    path += "lr{}_".format(config["learning_rate"])
    path += "opt-{}_".format(config["optim"])
    path += "loss-{}_".format(config["loss"])
    path += "splt{}_".format(config["split"])
    path += "s{}".format(config["seed"])

    return path


def get_data_loader(batch_size):

    train_data = np.load("data/train_data.npy").astype(np.float32)
    train_label = np.load("data/train_label.npy")
    val_data = np.load("data/val_data.npy").astype(np.float32)
    val_label = np.load("data/val_label.npy")

    train_dataset = GestureDataset(train_data, train_label)
    val_dataset = GestureDataset(val_data, val_label, add_noise=0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def split_data(split, s):

    instances = np.load("data/normalized_data.npy")
    labels = np.load("data/normalized_labels.npy")

    if split == 0:
        instances = np.delete(instances, 0, axis=2)

    elif split == 1:
        instances = np.delete(instances, 0, axis=2)
        instances = np.delete(instances, 2, axis=2)

    elif split == 2:
        instances = np.delete(instances, 0, axis=2)
        instances = np.delete(instances, 2, axis=2)
        instances = np.delete(instances, 2, axis=2)
        instances = np.delete(instances, 2, axis=2)
        instances = np.delete(instances, 2, axis=2)

    instances = np.swapaxes(instances, 1, 2)
    train_data, val_data, train_labels, val_labels = train_test_split(instances, labels,
                                                                      test_size=0.1, random_state=s)

    np.save("data/train_data.npy", train_data)
    np.save("data/val_data.npy", val_data)
    np.save("data/train_label.npy", train_labels)
    np.save("data/val_label.npy", val_labels)


def choose_loss(loss):

    if loss == "mse":
        return F.mse_loss, True
    elif loss == "bce":
        return F.binary_cross_entropy, True
    elif loss == "ce":
        return F.cross_entropy, False
    elif loss == "l1":
        return F.l1_loss, True


def choose_optimizer(optimizer):

    if optimizer == "adamax":
        return optim.Adamax
    elif optimizer == "adam":
        return optim.Adam


def make_model_directory(path):

    i = 0
    directory_created = False

    while not directory_created:
        try:
            os.mkdir(path+"_%d" % i)
            directory_created = True
        except FileExistsError:
            i += 1

    return path+"_%d" % i