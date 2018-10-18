'''
    Visualize some basic statistics of our dataset.
'''

import string
import random

import numpy as np
import matplotlib.pyplot as plt


def get_three_random_gestures():
    return random.sample(range(26), 3)


def find_average_sensor_values(gestures):

    gesture_means = []
    for letter_index in range(26):
        gesture = gestures.take(letter_index, axis=1)
        gesture_means.append(gesture.reshape(21500, 7).mean(axis=0)[1:7])

    return np.array(gesture_means)


def find_st_dev_sensor_values(gestures):

    gesture_st_dev = []
    for letter_index in range(26):
        gesture = gestures.take(letter_index, axis=1)
        gesture_st_dev.append(gesture.reshape(21500, 7).std(axis=0)[1:7])

    return np.array(gesture_st_dev)


def plot_bar_graphs(_means, _stds, gesture_indices):

    fig = plt.figure(figsize=(18*2.5, 6.5*2.5), dpi=80)

    for (i, gesture_index) in enumerate(gesture_indices):
        sub_fig = fig.add_subplot(1, 3, i+1)

        labels = ["x-acceleration", "y-acceleration", "z-acceleration", "pitch", "roll", "yaw"]
        sub_fig.bar(range(6), _means[gesture_index], yerr=_stds[gesture_index], tick_label=labels)
        sub_fig.tick_params(labelsize=14)
        sub_fig.set_title("Average Values for Gesture Representing Letter {}"
                          .format(string.ascii_lowercase[gesture_index]), fontsize=16)

    plt.suptitle("Some Average Values for Gestures over Time and Instances", fontsize=20)
    plt.show()


if __name__ == "__main__":

    instances = np.load("../data/instances.npy")
    means = find_average_sensor_values(instances)
    stds = find_st_dev_sensor_values(instances)

    plot_bar_graphs(means, stds, get_three_random_gestures())
