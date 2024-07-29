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


def circuit_training(X_train, Y_train, X_val, Y_val, num_qubits, num_layers, bias_init, learning_rate, batch_size,
                     num_epochs, optimizer=None, state_prep=None, seed=0):
    np.random.seed(seed)
    weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    weights = weights_init
    bias = bias_init
    all_accuracy = []
    if optimizer == 'Adam':
        opt = qml.AdamOptimizer(stepsize=learning_rate)
    elif optimizer == 'GDO':
        opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
    else:
        opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)

    for epoch in range(num_epochs):
        # Update the weights by one optimizer step, using only a limited batch of data
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = X_train[batch_index]
        Y_batch = Y_train[batch_index]
        weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch, num_qubits=num_qubits, state_prep=state_prep)

        # Compute predictions on train and validation set
        predictions_train = [np.sign(variational_classifier(weights, bias, x, num_qubits, state_prep)) for x in X_batch]

        predictions_val = [np.sign(variational_classifier(weights, bias, x, num_qubits, state_prep)) for x in X_val]
        current_cost = cost(weights, bias, X_batch, Y_batch, num_qubits, state_prep)

        # Compute accuracy on train and validation set
        acc_train = accuracy(Y_batch, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)

        print(f"Epoch: {epoch} | Cost: {current_cost:0.7f} | "f"Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}"
        )

    return weights

#%%
