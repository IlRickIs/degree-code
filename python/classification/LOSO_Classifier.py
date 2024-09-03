#this class uses the same classifiers as the Base_Classifier class but it uses the LOSO validation approach

from sklearn.pipeline import make_pipeline
from classification import helper
from sklearn.preprocessing import StandardScaler
import costants as C
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
import numpy as np
from classification.MetricsHandler import MetricsHandler
class LOSO_Classifier:
    def __init__(self, features, target, dataset_name):
        self.groups = features['actor']
        self.features = features.drop(columns=['actor'])
        self.target = target
        self.dataset_name = dataset_name

    def svm_classifier(self): 
        """Classify using SVM"""
        from sklearn.svm import SVC

        features = self.features
        loso = LeaveOneGroupOut()
        scaler = StandardScaler()
        columns = features.columns
        features = scaler.fit_transform(self.features)
        features = pd.DataFrame(features, columns=columns)
        
        #inizializza le liste per le metriche #TODO: refactor this by creating one object that contains all the metrics
        n_classes = len(np.unique(self.target))
        metrics_handler = MetricsHandler(self.dataset_name, 'LOSO_svm', n_classes)

        for train_idx, test_idx in loso.split(features, self.target, groups=self.groups):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = self.target.iloc[train_idx], self.target.iloc[test_idx]

            clf = make_pipeline(SVC())
            helper.optimize_svm_params(X_train, y_train, clf, self.dataset_name, C.PARAMS_LOSO_PATH)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            actor = self.groups.iloc[test_idx[0]]
            
            #aggiungi le metriche al report
            metrics_handler.add_actor_metrics(y_test, y_pred, actor)
        
        metrics_handler.print_big_report()