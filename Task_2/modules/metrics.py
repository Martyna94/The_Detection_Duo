from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support


def accuracy(labels, predictions):
    return accuracy_score(labels, predictions)


def auc(labels, predictions):
    return roc_auc_score(labels, predictions)


def precision_recall_fscore(labels, predictions):
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average='binary')


def show_all_metrics(labels, predictions):
    # Compute metrics
    acc = accuracy(labels, predictions)
    auc_score = auc(labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    print(f"Accuracy: {acc} | Precision: {precision} | Recall: {recall} | F1 Score: {fscore} | AUC-ROC: {auc_score}")


    # Visualize the confusion matrix
    cm, disp = confusion_metric(labels, predictions, display=True)
    disp.plot()


def confusion_metric(labels, predictions, display):
    cm = confusion_matrix(labels, predictions)
    if display:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        return cm, disp
    return cm
# %%
