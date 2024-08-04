import pennylane.numpy as np
import matplotlib.pyplot as plt


def create_data_sin_function(num_points, start=0, stop=2 * np.pi, test_split=0.2):
    """
    Generate training and test data based on the sine function over a specified range.

    Parameters:
        num_points (int): The number of input data points to generate.
        start (float): The starting value of the range for the input data. Default is 0.
        stop (float): The ending value of the range for the input data. Default is 2*pi.
        test_split (float): The proportion of data to be used for testing. Default is 0.2 (20%).

    Returns:
        tuple: Four numpy arrays containing the training and test input data points (X_train, X_test)
               and the corresponding sine values (Y_train, Y_test).

    Example:
        >>> X_train, Y_train, X_test, Y_test = create_data_sin_function(100)
    """
    X = np.linspace(start, stop, num_points)
    X.requires_grad = False
    Y = np.sin(X)

    indices = np.arange(num_points)
    np.random.shuffle(indices)

    num_test = int(np.floor(test_split * num_points))
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    X_train, Y_train = np.array(X[train_indices], requires_grad=False), np.array(Y[train_indices], requires_grad=False)
    X_test, Y_test = np.array(X[test_indices], requires_grad=False), np.array(Y[test_indices], requires_grad=False)

    print(f"Y_train: Length: {len(Y_train)}")
    print(f"Y_test: Length: {len(Y_test)}")

    return X_train, Y_train, X_test, Y_test


def create_data_noise_function(num_points, start=0, stop=2 * np.pi, test_split=0.2):
    """
    Generate training and test data based on the sine function over a specified range.

    Parameters:
        num_points (int): The number of input data points to generate.
        start (float): The starting value of the range for the input data. Default is 0.
        stop (float): The ending value of the range for the input data. Default is 2*pi.
        test_split (float): The proportion of data to be used for testing. Default is 0.2 (20%).

    Returns:
        tuple: Four numpy arrays containing the training and test input data points (X_train, X_test)
               and the corresponding sine values (Y_train, Y_test).

    Example:
        >>> X_train, Y_train, X_test, Y_test = create_data_sin_function(100)
    """
    X = np.linspace(-2 * np.pi, 2 * np.pi, num_points)
    X.requires_grad = False
    Y = np.sin(X) + 0.5 * np.sin(2 * X) + 0.25 * np.sin(3 * X)

    indices = np.arange(num_points)
    np.random.shuffle(indices)

    num_test = int(np.floor(test_split * num_points))
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    X_train, Y_train = np.array(X[train_indices], requires_grad=False), np.array(Y[train_indices], requires_grad=False)
    X_test, Y_test = np.array(X[test_indices], requires_grad=False), np.array(Y[test_indices], requires_grad=False)

    print(f"Y_train: Length: {len(Y_train)}")
    print(f"Y_test: Length: {len(Y_test)}")

    return X_train, Y_train, X_test, Y_test


def plot_result(X_test, Y_test, test_predictions, X_train=None, Y_train=None):
    """
    Plots the comparison of training data, test data, and predictions.

    Parameters:
    X_test (array-like): Input values for the test data.
    Y_test (array-like): Corresponding labels for the test data.
    test_predictions (array-like): Predicted values for the test data.
    X_train (array-like, optional): Input values for the training data. Default is None.
    Y_train (array-like, optional): Corresponding labels for the training data. Default is None.
    include_training (bool): If True, include training data in the plot.

    This function creates a scatter plot comparing the training data, test data,
    and the predictions made by the model. The x-axis
    is labeled with input values
    in radians (multiples of π), and the y-axis represents the corresponding labels.
    The plot includes:
    - Blue squares for training labels (if include_training is True)
    - Red circles for test labels
    - Black crosses for test predictions
    """
    fig, ax1 = plt.subplots()

    if X_train is not None and Y_train is not None:
        ax1.scatter(X_train, Y_train, s=30, c='b', marker='s', label='Train labels')

    ax1.scatter(X_test, Y_test, s=60, c='r', marker="o", label='Test labels')
    ax1.scatter(X_test, test_predictions, s=30, c='k', marker="x", label='Test predictions')

    ax1.set_xlabel("Inputs")
    ax1.set_ylabel("Labels")
    ax1.set_title("Comparison of Training and Test Data with Predictions")

    ax1.legend(loc='upper right')

    # Set x-axis ticks in radians
    x_ticks = np.linspace(min(X_test), max(X_test), num=5)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f"{x:.2f}π" for x in x_ticks / np.pi], rotation=45)

    plt.show()
# %%
