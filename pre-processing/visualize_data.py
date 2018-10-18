'''
    Visualize some samples.
'''

import string
import random

import numpy as np
import matplotlib.pyplot as plt


def get_random_gestures_indices():
    return int(random.random()*len(string.ascii_lowercase)),\
           [(int(random.random()*43), int(random.random()*5)) for _ in range(3)]


def get_instances(letter_index, student_number, instance_number):
    instances = np.load("../data/instances.npy")
    return instances[student_number][letter_index][instance_number]


def plot_gesture(gesture_instances, letter_index, gestures_indices):

    fig = plt.figure(figsize=(18*2.5, 6.5*2.5), dpi=80)

    for (i, ins) in enumerate(gesture_instances):
        sub_fig = fig.add_subplot(1, 3, i+1)

        sub_fig.set_title("Instance from student {}, instance number {}"
                          .format(gestures_indices[i][0], gestures_indices[i][1]), fontsize=16)

        t = ins[:, 0]
        sub_fig.plot(t, ins[:, 1], "r", label="x-acceleration")
        sub_fig.plot(t, ins[:, 2], "g", label="y-acceleration")
        sub_fig.plot(t, ins[:, 3], "b", label="z-acceleration")
        sub_fig.plot(t, ins[:, 4], "k", label="pitch")
        sub_fig.plot(t, ins[:, 5], "y", label="roll")
        sub_fig.plot(t, ins[:, 6], "c", label="yaw")

    plt.suptitle("Some Gesture Instances of Letter %s" % string.ascii_lowercase[letter_index], fontsize=20)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 0.2), fontsize='16')
    plt.show()


if __name__ == "__main__":

    letter_index_1, gestures_indices_1 = get_random_gestures_indices()
    letter_index_2, gestures_indices_2 = get_random_gestures_indices()

    gesture_1_instances = [get_instances(letter_index_1, *gestures_indices_1[i]) for i in range(3)]
    gesture_2_instances = [get_instances(letter_index_2, *gestures_indices_2[i]) for i in range(3)]

    plot_gesture(gesture_1_instances, letter_index_1, gestures_indices_1)
    plot_gesture(gesture_2_instances, letter_index_2, gestures_indices_2)
