import pennylane as qml


def layer(layer_weights, wires):
    qml.broadcast(qml.Rot, wires=wires, pattern='single', parameters=layer_weights)
    qml.broadcast(qml.CNOT, wires=wires, pattern='ring')


def state_preparation(x, wires, state_prep):
    if state_prep == 'Mottonen':
        qml.MottonenStatePreparation(state_vector=x, wires=wires)
    else:
        qml.BasisState(x, wires=wires)

dev = qml.device("default.qubit")
@qml.qnode(dev)
def circuit(weights, x, num_qubits, state_prep=None):
    wires = [i for i in range(num_qubits)]

    state_preparation(x, wires, state_prep)

    for layer_weights in weights:
        layer(layer_weights, wires)

    return qml.expval(qml.PauliZ(0))
