import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(loss_history, title="Training Loss"):

    plt.figure()

    plt.plot(loss_history)

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)

    plt.grid(True)

    plt.show()

def compute_accuracy(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = np.sum(y_true == y_pred)

    return correct / len(y_true)

def confusion_matrix(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN
    }

def compute_precision(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    TP = cm["TP"]
    FP = cm["FP"]

    return TP / (TP + FP) if (TP + FP) > 0 else 0

def compute_recall(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    TP = cm["TP"]
    FN = cm["FN"]

    return TP / (TP + FN) if (TP + FN) > 0 else 0

def compute_f1_score(y_true, y_pred):

    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def matthews_corrcoef(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    numerator = (TP * TN) - (FP * FN)

    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return numerator / denominator if denominator != 0 else 0




