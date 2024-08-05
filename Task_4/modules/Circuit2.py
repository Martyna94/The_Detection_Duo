import pennylane as qml

dev = qml.device('default.qubit', wires=1)

def S(x, scaling):
    qml.RX(scaling*x, wires=0)

def W(theta):
    qml.Rot(theta[0], theta[1], theta[2], wires=0)


@qml.qnode(dev)
def serial_quantum_model(weights, x, scaling=1):

    for theta in weights[:-1]:
        W(theta)
        S(x, scaling)

    # L+1'th unitary
    W(weights[-1])

    return qml.expval(qml.PauliZ(wires=0))