import pennylane.numpy as np
from .metrices import accuracy

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


def costfunc_exponential(params, X, Y, circuit, num_classes=6):
    """
    Compute the average exponential loss for a given dataset.

    Parameters:
    - params (array): Parameters used by the circuit function.
    - X (array): Input data array, where each element corresponds to a data sample.
    - Y (array): Array of integers representing the true labels for each data sample.
    - circuit (function): Function that computes probabilities for each class given input and parameters.
    - num_classes (int, optional): Number of classes in the classification task. Default is 6.

    Returns:
    - float: Average exponential loss over the dataset.

    Notes:
    - Assumes Y contains integers in the range [0, num_classes-1] representing class labels.
    - Assumes circuit(X[i], params) returns an array of probabilities for each class, indexed from 0 to num_classes-1.
    """
    len_X_set = len(X)
    Y_true_one_hot = np.eye(num_classes)[[i for i in range(num_classes)]]

    loss = 0.0

    for i in range(len_X_set):
        prob = circuit(X[i], params)

        if prob.ndim == 2:
            prob = prob[0]

        loss += (1+10*np.sum(np.sum(np.exp(7*np.sum(Y_true_one_hot[Y[i]] * prob[0:num_classes])))))**-1
    return loss


def costfunc_focal(params, X, Y, circuit, num_classes=6, gamma=2):
    """
    Compute the average focal loss for a given dataset.

    Parameters:
    - params (array): Parameters used by the circuit function.
    - X (array): Input data array, where each element corresponds to a data sample.
    - Y (array): Array of integers representing the true labels for each data sample.
    - circuit (function): Function that computes probabilities for each class given input and parameters.
    - num_classes (int, optional): Number of classes in the classification task. Default is 6.
    - gamma: value of hyperparameter gamma. Default is 2.

    Returns:
    - float: Average focal loss over the dataset.

    Notes:
    - Assumes Y contains integers in the range [0, num_classes-1] representing class labels.
    - Assumes circuit(X[i], params) returns an array of probabilities for each class, indexed from 0 to num_classes-1.
    """
    len_X_set = len(X)
    Y_true_one_hot = np.eye(num_classes)[[i for i in range(num_classes)]]

    loss = 0.0

    for i in range(len_X_set):
        prob = circuit(X[i], params)

        if prob.ndim == 2:
            prob = prob[0]

        p_i = np.sum(Y_true_one_hot[Y[i]] * prob[0:num_classes])

        loss -= ((1-p_i)**gamma)*np.log(p_i)

    return loss/len_X_set


def train_model(X_train, Y_train, X_val, Y_val, params, optimizer, circuit_peps, num_classes, num_epoch, batch_size):
    """
    Train a model using the specified parameters and optimizer.

    Parameters:
        X_train (ndarray): Training data features.
        Y_train (ndarray): Training data labels.
        X_val (ndarray): Validation data features.
        Y_val (ndarray): Validation data labels.
        params (dict): Initial model parameters.
        optimizer (Optimizer): Optimizer object with a step_and_cost method.
        circuit_peps (function): The quantum circuit or classical model to be used.
        num_classes (int): Number of classes in the classification task.
        num_epoch (int): Number of training epochs.
        batch_size (int): Size of each training batch.

    Returns:
        dict: Final parameters after training.
        list: Training accuracy over epochs.
        list: Validation accuracy over epochs.
        list: Costs over epochs.
        float: Total training time in hours.
    """

    all_params, train_accuracies, val_accuracies, costs = [], [], [], []
    all_params.append(params)

    for epoch in range(num_epoch):
        batch_index = np.random.randint(0, len(X_train), batch_size)
        X_batch = X_train[batch_index]
        Y_batch = Y_train[batch_index]

        params, cost = optimizer.step_and_cost(costfunc_cross_entropy, params, X=X_batch, Y=Y_batch, circuit=circuit_peps, num_classes=num_classes)

        if epoch == 0:
            costs.append(cost)

        current_cost = costfunc_cross_entropy(params, X=X_batch, Y=Y_batch, circuit=circuit_peps, num_classes=num_classes)

        acc_train = accuracy(params, X_batch, Y_batch, circuit_peps)
        acc_val = accuracy(params, X_val, Y_val, circuit_peps)

        print(f"Epoch: {epoch + 1} | Cost: {current_cost:0.7f} | Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}")

        train_accuracies.append(acc_train)
        val_accuracies.append(acc_val)
        all_params.append(params)
        costs.append(current_cost)

    print(params)


    return all_params, train_accuracies, val_accuracies, costs
