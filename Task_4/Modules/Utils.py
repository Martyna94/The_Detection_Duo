import pennylane.numpy as np


def create_data(num_points, test_split=0.2):
    """
    Generate training data based on the sine function over a specified number of points.

    Parameters:
        num_points (int): The number of input data points to generate from 0 to 2*pi.

    Returns:
        tuple: Two numpy arrays containing the input data points (X) and the corresponding sine values (Y).
    Example:
            >>> X_train, Y_train, X_test, Y_test = create_data(100)
    """
    X = np.linspace(0, 2 * np.pi, num_points)
    X.requires_grad = False
    Y = np.sin(X)

    indices = np.arange(num_points)
    np.random.shuffle(indices)

    num_test = int(np.floor(test_split * num_points))
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    X_train, Y_train = np.array(X[train_indices],requires_grad = False), np.array(Y[train_indices],requires_grad = False)
    X_test, Y_test = np.array(X[test_indices],requires_grad = False), np.array(Y[test_indices],requires_grad = False)

    return X_train, Y_train, X_test, Y_test


def evaluate_model(params, X_set, Y_set, circuit):
    """
    Evaluate the model on validation/test data and return the accuracy.

    Parameters:
        params: The parameters for the quantum circuit.
        X_set:  Set images.
        Y_set:  Set labels.
        circuit: The quantum circuit function that makes predictions.
    Returns:
        Accuracy percentage of the model on the validation/test set.
    """
    test_correct = 0
    test_predictions = []
    for img, true_label in zip(X_set, Y_set):
        prediction = circuit(img, params)
        test_predictions.append(prediction)

        if predicted_label == true_label:
            test_correct += 1

    accuracy_value = 100 * test_correct / len(X_set)
    return accuracy_value
#%%
