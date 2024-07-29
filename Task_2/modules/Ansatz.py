import pennylane as qml


def layer(layer_weights, wires):
    qml.broadcast(qml.Rot, wires=wires, pattern='single', parameters=layer_weights)
    qml.broadcast(qml.CNOT, wires=wires, pattern='ring')


def state_preparation(x, wires):
    qml.BasisState(x, wires=wires)


dev = qml.device("default.qubit")
@qml.qnode(dev)
def circuit(weights, x):
    wires = [i for i in range(len(x))]

    state_preparation(x, wires)

    for layer_weights in weights:
        layer(layer_weights, wires)

    return qml.expval(qml.PauliZ(0))
