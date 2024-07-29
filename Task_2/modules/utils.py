from pennylane import numpy as np
import itertools
from sklearn.model_selection import train_test_split


def load_data_4_qubit():
    data = np.loadtxt("./data/parity_4_qubit_train.txt", dtype=int)
    X_train = np.array(data[:, :-1])
    Y_train = np.array(data[:, -1])
    Y_train = Y_train * 2 - 1  # shift label from {0, 1} to {-1, 1}

    for x, y in zip(X_train, Y_train):
        print(f"x = {x}, y = {y}")
    data = np.loadtxt("./data/parity_4_qubit_test.txt", dtype=int)
    X_test = np.array(data[:, :-1])
    Y_test = np.array(data[:, -1])
    Y_test = Y_test * 2 - 1  # shift label from {0, 1} to {-1, 1}
    return X_train, Y_train, X_test, Y_test


def load_and_prepare_iris_data(train_ratio, val_ratio, test_ratio):
    if train_ratio + val_ratio + test_ratio > 1:
        raise ValueError("The sum of train_ratio and val_ratio cannot exceed 1.")

    data = np.loadtxt("./data/iris_classes1and2_scaled.txt")
    X = data[:, :4]
    Y = np.array(data[:, -1])
    X_norm = np.array([X[i] / np.linalg.norm(X[i])for i in range(len(X))])

    # First, split into training + validation and test
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X_norm, Y, test_size=test_ratio, shuffle=True, random_state=42)

    # Split training + validation into training and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=val_ratio_adjusted, shuffle=True, random_state=42)

    Y_train = np.array(Y_train)
    Y_val = np.array(Y_val)
    Y_test = np.array(Y_test)

    arrays = [Y_train, Y_val, Y_test]
    names = ["Y_train", "Y_val", "Y_test"]

    for y, name in zip(arrays, names):
        values, counts = np.unique(y, return_counts=True)
        print(f"{name}: Values: {values} Counts: {counts}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def generate_parity_dataset(num_bits, normalize = True):
    # Generate all possible binary strings of length num_bits
    X = np.array(list(itertools.product([0, 1], repeat=num_bits)), requires_grad=False)

    # Calculate the parity for each binary string
    Y = np.sum(X, axis=1) % 2
    if normalize:
        # normalize each input
        normalization = np.sqrt(np.sum(X**2, -1))
        X = (X.T / normalization).T
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    Y_train = Y_train * 2 - 1
    Y_test = Y_test * 2 - 1

    return X_train, X_test, Y_train, Y_test
