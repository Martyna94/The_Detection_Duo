import pennylane.numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

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
    predictions = []
    for img, true_label in zip(X_set, Y_set):
        prediction = circuit(img, params)
        predicted_label = np.argmax(prediction)
        predictions.append(predicted_label)

    return accuracy_score(Y_set, predictions)


def show_all_metrics(params, X_set, Y_set, circuit, target_names = ['class 0', 'class 1'], average= 'binary'):
    predictions = []
    for img, true_label in zip(X_set, Y_set):
        prediction = circuit(img, params)
        predicted_label = np.argmax(prediction)
        predictions.append(predicted_label)

    acc = accuracy_score(Y_set, predictions)
    auc_score = roc_auc_score(Y_set, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(Y_set, predictions, average='binary')
    balanced_acc = balanced_accuracy_score(Y_set, predictions)

    # Print the evaluation results in a structured format
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {fscore:.4f} | AUC-ROC: {auc_score:.4f} | Balanced Accuracy: {balanced_acc:.4f}")

    print(classification_report(Y_set, predictions, target_names=target_names))

    cm, disp = confusion_metric(Y_set, predictions, display=True)
    disp.plot()



def confusion_metric(labels, predictions, display):
    cm = confusion_matrix(labels, predictions)
    if display:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        return cm, disp
    return cm


    return
