import numpy as np
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, accuracy_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# If you ever have a binary task:
def binary_metrics(y_true, y_pred_labels):
    return {
        "precision": precision_score(y_true, y_pred_labels),
        "recall": recall_score(y_true, y_pred_labels),
        "f1": f1_score(y_true, y_pred_labels),
        "accuracy": accuracy_score(y_true, y_pred_labels),
    }
