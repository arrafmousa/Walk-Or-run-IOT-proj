import os
from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

import numpy as np


def count_entries_above_threshold(arr, entry_idx, thresholds):
    count = 0
    for sample in arr:
        if np.linalg.norm(sample[entry_idx]) > thresholds:
            count += 1
    return count


def create_sequential_dataset(directory):
    run_dataset = []  # List to store the loaded dataframes
    walk_dataset = []
    walk_labels = []
    run_labels = []
    # Iterate over files in the directory
    for filename in tqdm(os.listdir(directory)):
        try:
            is_walk = False
            if filename.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(directory, filename)
                f = open(file_path, "r")
                label = ''
                data = []
                for line in f.readlines():
                    items = line.split(",")
                    if items[0] == '"ACTIVITY TYPE:"':
                        is_walk = items[1].strip() == '"Running"'
                    if items[0] == '"COUNT OF ACTUAL STEPS:"':
                        label = [float(items[1].replace("\"", "").strip())]
                    if len(items) == 4 and items[0] != '"Time [sec]"':
                        data.append([float(item.replace("\"", "").strip()) for item in items])
                if is_walk:
                    walk_dataset.append(data)
                    walk_labels.append(label)
                else:
                    run_dataset.append(data)
                    run_labels.append(label)

        except:
            pass

    return [walk_dataset, run_dataset], [walk_labels, run_labels]


new_dataset, labels = create_sequential_dataset(r"dataIOT")


def get_threshold(idx):
    best_threshold = -100
    closest_to_traget = 10000
    for x_threshold in tqdm([x/100 for x in range(100,2000)]):
        error_for_threshold = 0
        for example, target in zip(new_dataset[0], labels[0]):
            np_example = np.array(example)
            predicted = count_entries_above_threshold(np_example, idx, x_threshold)
            error_for_threshold += predicted - target[0]
        error_for_threshold = error_for_threshold / len(labels)
        if abs(error_for_threshold) < closest_to_traget:
            closest_to_traget = abs(error_for_threshold)
            best_threshold = (x_threshold)

    print(f"best {idx} : {best_threshold}")
    print(f"best accuracy : {closest_to_traget}")
    return best_threshold


get_threshold([1, 2, 3])
