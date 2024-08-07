import pennylane.numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
from .training import training, cost_MSE


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


def create_data_advance_function(num_points, coeff0, coeffs,start=0, stop=2 * np.pi, scaling=1):
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
        >>> X_train, Y_train = create_data_sin_function(100)
    """
    X_train = np.linspace(start, stop, num_points)
    X_train.requires_grad = False
    Y_train= np.array([target_function(x_,coeff0=coeff0,coeffs=coeffs,scaling=scaling) for x_ in X_train],requires_grad = False)

    plot_result("Training Data Visualization", X_train=X_train, Y_train=Y_train)

    return X_train, Y_train

def target_function(x,coeff0,coeffs,scaling):
    """Generate a truncated Fourier series of degree, where the data gets re-scaled."""
    res = coeff0
    for idx, coeff in enumerate(coeffs):
        exponent = complex(0, scaling*(idx+1)*x)
        conj_coeff = np.conjugate(coeff)
        res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
    return np.real(res)


def plot_result(title, X_test=None, Y_test=None, test_label="Test labels", test_predictions=None, X_train=None, Y_train=None, shuffle = False):
    """
    Plots the comparison of training data, test data, and predictions.

    Parameters:
    X_test (array-like, optional): Input values for the test data. Default is None.
    Y_test (array-like, optional): Corresponding labels for the test data. Default is None.
    test_predictions (array-like, optional): Predicted values for the test data. Default is None.
    X_train (array-like, optional): Input values for the training data. Default is None.
    Y_train (array-like, optional): Corresponding labels for the training data. Default is None.

    This function creates a scatter plot comparing the training data, test data,
    and the predictions made by the model. If only training data is provided,
    it plots the training data. The x-axis is labeled with input values in radians (multiples of π),
    and the y-axis represents the corresponding labels.
    """
    sns.set(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    if X_train is not None and Y_train is not None:
        if not shuffle:
            ax1.plot(X_train, Y_train, c='black', label='Train Data')
        ax1.scatter(X_train, Y_train, s=60, facecolor='white', edgecolor='black', label='Train Points', alpha=0.6)

    if X_test is not None and Y_test is not None:
        ax1.scatter(X_test, Y_test, s=80, c='blue', label=test_label, alpha=0.8, edgecolor='k')

    if test_predictions is not None:
        ax1.scatter(X_test, test_predictions, s=60, c='red', marker="x", label='Test predictions')

    ax1.set_xlabel("Inputs (radians)", fontsize=14)
    ax1.set_ylabel("Labels", fontsize=14)
    ax1.set_title(title, fontsize=16, weight='bold')
    ax1.legend(loc='upper right', fontsize=12)

    # Set x-axis ticks in radians if X_test or X_train is available
    if X_test is not None:
        x_ticks = np.linspace(min(X_test), max(X_test), num=5)
    elif X_train is not None:
        x_ticks = np.linspace(min(X_train), max(X_train), num=5)
    else:
        x_ticks = []
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f"{x:.2f}π" for x in x_ticks / np.pi], rotation=45, fontsize=12)

    ax1.tick_params(axis='y', labelsize=12)

    plt.ylim(-1, 1)
    plt.show()


def perform_training(num_points_array):
    num_epochs = 30
    stepsize = 0.1
    init_params = [0.1, 0.1]
    summary = []
    # Generate test set, size always the same not depends on params `num_points`
    X_test, Y_test,_,_ = create_data_sin_function(1000,start= 2 * np.pi,stop = 7 * np.pi)

    for num_points in num_points_array:
        # Generate training
        X_train = np.linspace(0, 2 * np.pi, num_points)
        X_train.requires_grad = False
        Y_train = np.sin(X_train)

        # Initialize optimizer and parameters
        opt = qml.GradientDescentOptimizer(stepsize=stepsize)
        init_params = np.array(init_params, requires_grad=True)

        # Perform training
        final_params, costs = training(num_epochs, opt, cost_MSE, init_params, X_train, Y_train)

        test_cost = cost_MSE(final_params, X_test, Y_test)
        summary.append((num_points, final_params, costs[-1],test_cost))

        print(f"Training completed with {num_points} data points.")
        print(f"Final parameters: {final_params}")
        print(f"Final cost: {costs[-1]:0.7f}")
        print(f"Test cost: {test_cost:0.7f}")
    # Print summary of all results
    print("\nSummary of all training runs:")
    for num_points, final_params, final_cost, test_cost in summary:
        print(f"Training completed with {num_points} data points | Final parameters: {final_params} | Final cost: {final_cost:0.7f}| Test cost: {test_cost:0.7f}")


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

        # Add annotations for max and min values
        max_value = max(metric_values)
        max_index = metric_values.index(max_value)
        min_value = min(metric_values)
        min_index = metric_values.index(min_value)

        plt.annotate(f'Max: {max_value:.8f}', (max_index, max_value), textcoords="offset points", xytext=(50, 0), ha='center', color='red')
        plt.scatter(max_index, max_value, color='red', s=100)

        plt.annotate(f'Min: {min_value:.8f}', (min_index, min_value), textcoords="offset points", xytext=(0, 15), ha='center', color='green')
        plt.scatter(min_index, min_value, color='green', s=100)

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

# %%
