'''
    Save the data in the .csv file, save as a .npy file in ./data
'''

import numpy as np
import string


def load_instance(student_number, letter, instance_number):
    path = "../raw_data/student%d/%s_%d.csv" % (student_number, letter, instance_number)
    return np.genfromtxt(path, delimiter=',')


def load_data():

    labels = []
    instances = []

    for student_number in range(43):
        instances_per_student = []
        labels_per_student = []
        for letter in string.ascii_lowercase:
            instances_per_letter = []
            labels_per_letter = []
            for instance_number in range(1, 6):
                instances_per_letter.append(load_instance(student_number, letter, instance_number))
                labels_per_letter.append(letter)
            labels_per_student.append(labels_per_letter)
            instances_per_student.append(instances_per_letter)
        labels.append(labels_per_student)
        instances.append(instances_per_student)

    return np.array(labels), np.array(instances)


if __name__ == "__main__":
    gesture_labels, gesture_instances = load_data()

    np.save("../data/labels.npy", gesture_labels)
    np.save("../data/instances.npy", gesture_instances)
