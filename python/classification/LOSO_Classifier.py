#this class uses the same classifiers as the Base_Classifier class but it uses the LOSO validation approach

from sklearn.pipeline import make_pipeline
from classification import helper
from sklearn.preprocessing import StandardScaler
import costants as C
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd
class LOSO_Classifier:
    def __init__(self, features, target, dataset_name):
        self.groups = features['actor']
        self.features = features.drop(columns=['actor'])
        self.target = target
        self.dataset_name = dataset_name

    def __single_actors_report(self, actors, accuracies, precisions, recalls, f1_scores, filename):
        """Create a report for each actor"""
        if not os.path.exists(C.SINGLE_ACTOR_REPORTS_PATH):
            os.makedirs(C.SINGLE_ACTOR_REPORTS_PATH)

        with open(filename, 'w') as f:
            for i in range(len(actors)):
                f.write(f'Actor: {actors[i]}\n')
                f.write(f'Accuracy: {accuracies[i]}\n')
                f.write(f'Precision: {precisions[i]}\n')
                f.write(f'Recall: {recalls[i]}\n')
                f.write(f'F1: {f1_scores[i]}\n\n')

    def svm_classifier(self): #TODO: sometimes an actor get 1 in all the metrics, is not possible, correct this
        """Classify using SVM"""
        from sklearn.svm import SVC

        features = self.features
        loso = LeaveOneGroupOut()
        scaler = StandardScaler()
        columns = features.columns
        features = scaler.fit_transform(self.features)
        features = pd.DataFrame(features, columns=columns)
        

        actors = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for train_idx, test_idx in loso.split(features, self.target, groups=self.groups):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = self.target.iloc[train_idx], self.target.iloc[test_idx]

            clf = make_pipeline(SVC())
            helper.optimize_svm_params(X_train, y_train, clf, self.dataset_name, C.PARAMS_LOSO_PATH)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            print('test - pred: ', set(y_test) - set(y_pred))
            # Calcola le metriche macro per multi-classe
            precision = precision_score(y_test, y_pred, average='macro')
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            actor = self.groups.iloc[test_idx[0]]
            
            actors.append(actor)
            precisions.append(precision)
            accuracies.append(accuracy)
            recalls.append(recall)
            f1_scores.append(f1)
        
        self.__single_actors_report(actors, 
                                    accuracies, 
                                    precisions, 
                                    recalls, 
                                    f1_scores, 
                                    C.SINGLE_ACTOR_REPORTS_PATH + self.dataset_name + '_svm_report.txt')
        #TODO: create a fnc to calculate the average of the metrics
        print('Average metrics:')
        report = f'Average accuracy: {sum(accuracies)/len(accuracies)}\n\
                    Average precision: {sum(precisions)/len(precisions)}\n\
                    Average recall: {sum(recalls)/len(recalls)}\n\
                    Average F1: {sum(f1_scores)/len(f1_scores)}\n'
        print(report)
