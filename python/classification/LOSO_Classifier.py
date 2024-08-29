#this class uses the same classifiers as the Base_Classifier class but it uses the LOSO validation approach

from sklearn.pipeline import make_pipeline
from classification import helper
from sklearn.preprocessing import StandardScaler
import costants as C
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LOSO_Classifier:
    def __init__(self, features, target, dataset_name):
        self.features = features
        self.groups = features['actor']
        self.features = self.features.drop(columns=['actor'])
        self.target = target
        self.dataset_name = dataset_name

    def __print_report(self,y_test, y_pred, classifier_name):
        report = classification_report(y_test, y_pred, labels=[1, 3, 4, 5], target_names=['neu', 'happy', 'sad', 'ang'])
        confusion = confusion_matrix(y_test, y_pred, labels=[1, 3, 4, 5], normalize=None)

        print(report)
        print()
        print(confusion)
        print()
        report_dict = classification_report(y_test, y_pred, labels=[1, 3, 4, 5], target_names=['neu', 'happy', 'sad', 'ang'], output_dict=True)
        helper.write_report(report_dict, self.dataset_name, classifier_name, latex=True)
        helper.write_cool_confusion_matrix(confusion,['neu', 'happy', 'sad', 'ang'], self.dataset_name, classifier_name)
    
    def svm_classifier(self): #TODO: sometimes an actor get 1 in all the metrics, is not possible, correct this
        """Classify using SVM"""
        from sklearn.svm import SVC

        loso = LeaveOneGroupOut()
        
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for train_idx, test_idx in loso.split(self.features, self.target, groups=self.groups):
            X_train, X_test = self.features.iloc[train_idx], self.features.iloc[test_idx]
            y_train, y_test = self.target.iloc[train_idx], self.target.iloc[test_idx]

            clf = make_pipeline(StandardScaler(), SVC())
            params = helper.optimize_svm_params(X_train, y_train, clf, self.dataset_name)
            clf.set_params(**params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
    
            # Calcola le metriche macro per multi-classe
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

            report = f'Actor {self.groups.iloc[test_idx[0]]}\n\
                      Accuracy: {acc}\n\
                      Precision: {precision}\n\
                      Recall: {recall}\n\
                      F1: {f1}\n'
            
            print(report)

        #TODO: create a fnc to calculate the average of the metrics
        print('Average metrics:')
        report = f'Average accuracy: {sum(accuracies)/len(accuracies)}\n\
                    Average precision: {sum(precisions)/len(precisions)}\n\
                    Average recall: {sum(recalls)/len(recalls)}\n\
                    Average F1: {sum(f1_scores)/len(f1_scores)}\n'
        print(report)