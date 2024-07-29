import pennylane as qml
from pennylane import numpy as np
from .Ansatz import circuit
from .metrics import accuracy


def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias


def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)


def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)


def circuit_training(X_train, Y_train, num_qubits, num_layers, bias_init, learning_rate, batch_size,
                     num_epochs, optimizer=None, seed=0):
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
        weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch)

        # Compute accuracy
        predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X_batch]

        current_cost = cost(weights, bias, X_batch, Y_batch)
        acc = accuracy(Y_batch, predictions)

        print(f"Iter: {epoch + 1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")

    return weights
