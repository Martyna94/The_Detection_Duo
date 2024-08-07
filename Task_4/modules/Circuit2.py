import pennylane as qml

dev = qml.device('default.qubit', wires=1)

def S(x, scaling):
    qml.RX(scaling*x, wires=0)

def W(theta):
    qml.Rot(theta[0], theta[1], theta[2], wires=0)


def layer(layer_weights,x, scaling):
    for theta in layer_weights:
        W(theta)
        S(x, scaling)

@qml.qnode(dev)
def quantum_circuit_2(x,weights, scaling=1):

    for layer_weights in weights[:-1]:
        layer(layer_weights, x, scaling)
    # L+1'th unitary
    W(*weights[-1])

    return qml.expval(qml.PauliZ(wires=0))