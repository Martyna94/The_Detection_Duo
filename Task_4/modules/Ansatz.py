import pennylane as qml

dev = qml.device('default.qubit', wires=1)

@qml.qnode(dev)
def quantum_circuit(x, weights):
    # Encoding
    qml.AngleEmbedding(x, wires=[0])
    # Ansatz
    qml.RX(weights[0], wires=[0])
    qml.RX(weights[1], wires=[0])
    return qml.expval(qml.PauliZ(wires=0))
#%%
