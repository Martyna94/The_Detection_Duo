from pennylane import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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


def plot_scatter(X, Y, dim1, dim2, title=None):
    plt.figure()
    plt.scatter(X[:, dim1][Y == 1], X[:, dim2][Y == 1], c="b", marker="o", ec="k", label="Class 1")
    plt.scatter(X[:, dim1][Y == -1], X[:, dim2][Y == -1], c="r", marker="o", ec="k", label="Class -1")
    plt.xlabel(f"Dimension {dim1}")
    plt.ylabel(f"Dimension {dim2}")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def plot_metrics_accuracy_and_cost(all_accuracy, all_cost):
    """
    Plot accuracy and cost over epochs.

    Parameters:
    - all_accuracy (list): List of accuracy values over epochs.
    - all_cost (list): List of cost values over epochs.
    """
    plt.figure(figsize=(20, 5))

    # Plot 1: Accuracy over Epochs
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, plot 1
    plt.plot(all_accuracy, marker='o', linestyle='-', color='b')
    plt.title('Accuracy Value over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Value')
    plt.ylim(0, 100)  # Set the y-axis to range from 0 to 100
    plt.grid(True)

    # Calculate the maximum accuracy value and its index
    max_accuracy = max(all_accuracy)
    max_index = all_accuracy.index(max_accuracy)

    # Annotate the maximum accuracy value
    plt.annotate(f'Max: {max_accuracy:.2f}%', (max_index, max_accuracy), textcoords="offset points", xytext=(0, 10), ha='center', color='red')

    # Mark the maximum accuracy value
    plt.scatter(max_index, max_accuracy, color='red', s=100)

    # Plot 2: Cost over Epochs
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, plot 2
    plt.plot(all_cost, marker='o', linestyle='-', color='b')
    plt.title('Cost Function Value over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cost Value')
    plt.ylim(bottom=0)  # Set the y-axis to start from 0
    plt.grid(True)

    # Display the plots
    plt.show()


def plot_metrics_over_epochs(metrics_dict):
    """
    Plot metrics over epochs.

    Parameters:
    - metrics_dict (dict): Dictionary where keys are metric names and values are lists of metric values over epochs.
    """
    num_metrics = len(metrics_dict)
    plt.figure(figsize=(6 * num_metrics, 5))

    # Plot each metric
    for i, (metric_name, metric_values) in enumerate(metrics_dict.items(), start=1):
        plt.subplot(1, num_metrics, i)
        plt.plot(metric_values, marker='o', linestyle='-', color='b')
        plt.title(f'{metric_name} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric_name} Value')
        plt.grid(True)

        # Add annotations or customizations as needed
        if metric_name == 'Accuracy':
            plt.ylim(0, 100)  # Set y-axis limit for accuracy
            max_accuracy = max(metric_values)
            max_index = metric_values.index(max_accuracy)
            plt.annotate(f'Max: {max_accuracy:.2f}%', (max_index, max_accuracy), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
            plt.scatter(max_index, max_accuracy, color='red', s=100)
        elif metric_name == 'Cost':
            plt.ylim(bottom=0)  # Set y-axis start from 0 for cost

    # Adjust layout and display plots
    plt.tight_layout()
    plt.show()