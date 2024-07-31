import pennylane.numpy as np


def costfunc_cross_entropy(params, X, Y, circuit, num_classes=6):
    """
    Compute the average cross-entropy loss for a given dataset.

    Parameters:
    - params (array): Parameters used by the circuit function.
    - X (array): Input data array, where each element corresponds to a data sample.
    - Y (array): Array of integers representing the true labels for each data sample.
    - circuit (function): Function that computes probabilities for each class given input and parameters.
    - num_classes (int, optional): Number of classes in the classification task. Default is 6.

    Returns:
    - float: Average cross-entropy loss over the dataset.

    Notes:
    - Assumes Y contains integers in the range [0, num_classes-1] representing class labels.
    - Assumes circuit(X[i], params) returns an array of probabilities for each class, indexed from 0 to num_classes-1.
    """
    len_X_set = len(X)
    Y_true_one_hot =  np.eye(num_classes)[[i for i in range(num_classes)]]

    loss = 0.0

    for i in range(len_X_set):
        prob = circuit(X[i], params)

        if prob.ndim == 2:
            prob = prob[0]

        loss -= np.sum(Y_true_one_hot[Y[i]] * np.log(prob[0:num_classes]))

    return loss/len_X_set

