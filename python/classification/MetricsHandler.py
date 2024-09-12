from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import costants as c
from classification import helper
class MetricsHandler:
    def __init__(self, dataset_name, classif_name, n_classes):
        """Initialize the metrics handler\n
        filename: str, the name of the file to write the metrics to,
        for convention use the format: dataset_name_classifier_name"""
        self.filename = c.REPORTS_LOSO_PATH + dataset_name + '_' + classif_name + '_report.txt'
        self.dataset_name = dataset_name
        self.classifier_name = classif_name
        self.actors = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.never_predicted = []
        self.cumulative_cm = np.zeros((n_classes, n_classes), dtype=int)
    
    def add_actor_metrics(self, y_true, y_pred, actor):
        """Calculate metrics for each actor""" 
        self.actors.append(actor)
        self.accuracies.append(accuracy_score(y_true, y_pred))
        self.precisions.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
        self.recalls.append(recall_score(y_true, y_pred, average='macro', zero_division=0))
        self.f1_scores.append(f1_score(y_true, y_pred, average='macro', zero_division=0))
        self.never_predicted.append(list(set(y_true) - set(y_pred)))
        self.cumulative_cm += confusion_matrix(y_true, y_pred)
    
    def get_metrics(self):
        """Return metrics as a dictionary"""
        return {'actors': self.actors,
                'accuracies': self.accuracies,
                'precisions': self.precisions,
                'recalls': self.recalls,
                'f1_scores': self.f1_scores,
                'never_predicted': self.never_predicted,
                'cumulative_cm': self.cumulative_cm}
    
    def print_big_report(self):
        """Create a report for each actor, and the average metrics"""
        path = os.path.dirname(self.filename)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(self.filename, 'w') as f:
            for i in range(len(self.actors)):
                f.write(f'Actor: {self.actors[i]}\n')
                f.write(f'Accuracy: {self.accuracies[i]}\n')
                f.write(f'Precision: {self.precisions[i]}\n')
                f.write(f'Recall: {self.recalls[i]}\n')
                f.write(f'F1: {self.f1_scores[i]}\n')
                f.write(f'Never predicted labels for this actor: {self.never_predicted[i]}\n\n')

            report = (
                f"\n---| Average metrics: |---\n"
                f"Average accuracy: {np.mean(self.accuracies):.4f}\n"
                f"Average precision: {np.mean(self.precisions):.4f}\n"
                f"Average recall: {np.mean(self.recalls):.4f}\n"
                f"Average F1: {np.mean(self.f1_scores):.4f}\n"
                f"Never predicted labels: {', '.join(map(str, set([item for sublist in self.never_predicted for item in sublist])))}\n"
            )
            print(report)
            f.write(report)
            
        if(c.NORMALIZE_MATRIX == 'true'):
            row_sums = self.cumulative_cm.sum(axis=1, keepdims=True)
            self.cumulative_cm = self.cumulative_cm / row_sums

        helper.write_cool_confusion_matrix(self.cumulative_cm, 
                                            ['neu', 'happy', 'sad', 'ang'], 
                                            self.dataset_name, self.classifier_name)
        print(self.cumulative_cm)
        