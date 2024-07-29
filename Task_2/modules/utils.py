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


def generate_parity_dataset(num_bits):
    # Generate all possible binary strings of length num_bits
    X = np.array(list(itertools.product([0, 1], repeat=num_bits)), requires_grad=False)

    # Calculate the parity for each binary string
    Y = np.sum(X, axis=1) % 2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    # Y_train = Y_train * 2 - 1
    # Y_test = Y_test * 2 - 1

    return X_train, X_test, Y_train, Y_test
