import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit")
@qml.qnode(dev)
def circuit_centric(weights, x, num_qubits=4):
    wires = [i for i in range(num_qubits)]
    qml.AmplitudeEmbedding(features=x, wires=wires, normalize=True)
    qml.StronglyEntanglingLayers(weights=weights, wires=range(4))
    return qml.expval(qml.Z(0))


def variational_classifier(weights, bias, x):
    pi = (circuit_centric(weights, x)/2 + 1/2)
    return  pi + bias


def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return 1/2 * np.sum((qml.math.stack(predictions)-labels) ** 2)


def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) > 0 for x in X]
    return square_loss(Y, predictions)
