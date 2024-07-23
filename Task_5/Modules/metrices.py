import pennylane.numpy as np


def accuracy(params, X_set, Y_set, circuit):
    """
    Evaluate the model on validation/test data and return the accuracy.

    Parameters:
        params: The parameters for the quantum circuit.
        X_set:  Set images.
        Y_set:  Set labels.
        circuit: The quantum circuit function that makes predictions.
    Returns:
        Accuracy percentage of the model on the validation/test set.
    """
    test_correct = 0
    for img, true_label in zip(X_set, Y_set):
        predictions = circuit(img, params)
        predicted_label = np.argmax(predictions)

        if predicted_label == true_label:
            test_correct += 1

    accuracy_value = 100 * test_correct / len(X_set)

    return accuracy_value

def F1_score():
    return
