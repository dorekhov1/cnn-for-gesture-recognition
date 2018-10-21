'''
    Normalize the data, save as ./data/normalized_data.npy
'''

import numpy as np


def normalize_instances_locally(data):

    normalized_data = []

    for student_index in range(43):
        data_per_student = []
        for letter_index in range(26):
            data_per_label = []
            for instance_index in range(5):
                instance = data[student_index][letter_index][instance_index]
                data_per_label.append((instance - instance.mean(axis=0))/instance.std(axis=0))
            data_per_student.append(data_per_label)
        normalized_data.append(data_per_student)

    return np.array(normalized_data)


def normalize_data_locally(data):

    normalized_data = []

    for d in range(1170):
        instance = data[d]
        normalized_data.append((instance - instance.mean(axis=0))/instance.std(axis=0))

    return np.array(normalized_data)


def convert_to_ints(_labels):
    return np.array([ord(label) - 97 for label in _labels])


if __name__ == "__main__":

    gesture_instances = np.load("../data/instances.npy")
    normalized_gesture_instances = normalize_instances_locally(gesture_instances).reshape((5590, 100, 7))
    np.save("../data/normalized_data.npy", normalized_gesture_instances)

    labels = np.load("../data/labels.npy").reshape(5590)

    labels = convert_to_ints(labels)
    np.save("../data/normalized_labels.npy", labels)

    test_data = np.load("../data/test_data.npy")
    normalized_test_data = normalize_data_locally(test_data)
    np.save("../data/normalized_test.npy", normalized_test_data)
