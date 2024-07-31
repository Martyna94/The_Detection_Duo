import pennylane as qml
from pennylane import numpy as np
from .Ansatz import circuit
from .metrics import accuracy


def variational_classifier(weights, bias, x, num_qubits, state_prep):
    return circuit(weights, x, num_qubits, state_prep) + bias


def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)


def cost(weights, bias, X, Y, num_qubits, state_prep):
    predictions = [variational_classifier(weights, bias, x, num_qubits, state_prep) for x in X]
    return square_loss(Y, predictions)


def initialize_weights(num_layers, num_qubits, seed):
    np.random.seed(seed)
    return 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)


def initialize_optimizer(optimizer_name, learning_rate):
    if optimizer_name == 'Adam':
        return qml.AdamOptimizer(stepsize=learning_rate)
    elif optimizer_name == 'GDO':
        return qml.GradientDescentOptimizer(stepsize=learning_rate)
    else:
        return qml.NesterovMomentumOptimizer(stepsize=learning_rate)


def get_batch(X_train, Y_train, batch_size):
    batch_index = np.random.randint(0, len(X_train), (batch_size,))
    return X_train[batch_index], Y_train[batch_index]


def compute_predictions(weights, bias, X, num_qubits, state_prep):
    return [np.sign(variational_classifier(weights, bias, x, num_qubits, state_prep)) for x in X]


def circuit_training(X_train, Y_train, X_val, Y_val, num_qubits, num_layers, learning_rate, batch_size,
                     num_epochs, optimizer=None, state_prep=None, seed=0):

    weights_init = initialize_weights(num_layers, num_qubits, seed)
    bias_init = np.array(0.0, requires_grad=True)

    weights = weights_init
    bias = bias_init

    train_accuracies, val_accuracies, costs, biases = [], [], [], []

    opt = initialize_optimizer(optimizer, learning_rate)

    for epoch in range(num_epochs):
        X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)

        weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch, num_qubits=num_qubits, state_prep=state_prep)

        predictions_train = compute_predictions(weights, bias, X_batch, num_qubits, state_prep)
        predictions_val = compute_predictions(weights, bias, X_val, num_qubits, state_prep)

        current_cost = cost(weights, bias, X_batch, Y_batch, num_qubits, state_prep)

        acc_train = accuracy(Y_batch, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)

        print(f"Epoch: {epoch} | Cost: {current_cost:0.7f} | "f"Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}")

        costs.append(current_cost)
        train_accuracies.append(acc_train)
        val_accuracies.append(acc_val)
        biases.append(bias)

    return weights, costs, train_accuracies, val_accuracies, biases

def circuit_training_centric_ansatz(X_train, Y_train, X_val, Y_val, num_qubits, num_layers, learning_rate, batch_size,
                     num_epochs, optimizer=None, seed=0):

    weights_init = initialize_weights(num_layers, num_qubits, seed)
    bias_init = np.array(0.0, requires_grad=True)

    weights = weights_init
    bias = bias_init

    train_accuracies, val_accuracies, costs, biases = [], [], [], []

    opt = initialize_optimizer(optimizer, learning_rate)

    for epoch in range(num_epochs):
        X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)

        weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch, num_qubits=num_qubits, state_prep=state_prep)

        predictions_train = compute_predictions(weights, bias, X_batch, num_qubits, state_prep)
        predictions_val = compute_predictions(weights, bias, X_val, num_qubits, state_prep)

        current_cost = cost(weights, bias, X_batch, Y_batch, num_qubits, state_prep)

        acc_train = accuracy(Y_batch, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)

        print(f"Epoch: {epoch} | Cost: {current_cost:0.7f} | "f"Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}")

        costs.append(current_cost)
        train_accuracies.append(acc_train)
        val_accuracies.append(acc_val)
        biases.append(bias)

    return weights, costs, train_accuracies, val_accuracies, biases

#%%
