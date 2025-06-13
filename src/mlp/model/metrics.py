import numpy as np

class Metrics:
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,):
        self.y_true = y_true
        self.y_pred = y_pred
        self.TP = np.sum((y_true == 1) & (y_pred == 1))
        self.TN = np.sum((y_true == 0) & (y_pred == 0))
        self.FP = np.sum((y_true == 0) & (y_pred == 1))
        self.FN = np.sum((y_true == 1) & (y_pred == 0))

    def _calculate_accuracy(self) -> float:
        """
            Calculate the accuracy of the model on test data : global perfomance.
            (TP + TN) / Total

            :param y_pred: Calculated predictions
            :param y_true: True labels (Malignant or Benign)
            :return: accuracy score (0-1)
        """
        return (self.TP + self.TN) / (self.TP + self.FP + self.TN + self.FN) # epoch_accuracy = np.mean(y_pred_train == y_true_binary)

    def _calculate_precision(self) -> float:
        """
            Calculate the precision of the model on test data.
            TP / (TP + FP)
            :return: precision score (0-1)
        """
        if (self.TP + self.FP) == 0:
            return 0.0
        return self.TP / (self.TP + self.FP)

    def _calculate_recall(self) -> float:
        """
            Calculate the recall of the model on test data.
            TP / (TP + FN)
            :return: recall score (0-1)
        """
        if (self.TP + self.FN) == 0:
            return 0.0
        return self.TP / (self.TP + self.FN)

    def _calculate_F1score(self) -> float:
        """
            Calculate the F1 score of the model on test data.
            f1 = 2 * (precision * recall) / (precision + recall)
            :return: F1 score (0-1)
        """
        precision = self._calculate_precision()
        recall = self._calculate_recall()
        if (precision * recall) / (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def print_all_metrics(self):
        """
            Print calculated metrics for the prediction phase.
        """
        accuracy = self._calculate_accuracy()
        precision = self._calculate_precision()
        recall = self._calculate_recall()
        f1_score = self._calculate_F1score()
        specificity = self.TN / (self.TN + self.FP) if (self.TN + self.FP) > 0 else 0.0
        
        print("\n" + "="*40)
        print("EVALUATION METRICS")
        print("="*40)
        
        print(f"Accuracy    : {accuracy:.4f}")
        print(f"Precision   : {precision:.4f}")
        print(f"Recall      : {recall:.4f}")
        print(f"Specificity : {specificity:.4f}")
        print(f"F1-Score    : {f1_score:.4f}")

        print(f"\nTP: {self.TP}, TN: {self.TN}, FP: {self.FP}, FN: {self.FN}")
        print("="*40 + "\n")