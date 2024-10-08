#this class uses the same classifiers as the Base_Classifier class but it uses the LOSO validation approach
from sklearn.pipeline import make_pipeline
from classification import helper
from sklearn.preprocessing import StandardScaler
import costants as c
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
import numpy as np
from classification.MetricsHandler import MetricsHandler
class LosoClassifier:
    def __init__(self, features, target, dataset_name, sample_weights=None, optimization = True):
        self.groups = features['actor']
        self.features = features.drop(columns=['actor'])
        self.target = target
        self.dataset_name = dataset_name
        self.sample_weights = sample_weights
        self.optimization = optimization

    def svm_classifier(self): 
        """Classify using SVM"""
        from sklearn.svm import SVC

        features = self.features
        loso = LeaveOneGroupOut()
        scaler = StandardScaler()
        columns = features.columns
        features = scaler.fit_transform(self.features)
        features = pd.DataFrame(features, columns=columns)
        
        #inizializza le liste per le metriche 
        n_classes = len(np.unique(self.target))
        metrics_handler = MetricsHandler(self.dataset_name, 'LOSO_svm', n_classes)

        for train_idx, test_idx in loso.split(features, self.target, groups=self.groups):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = self.target.iloc[train_idx], self.target.iloc[test_idx]
            
            sample_weight = (
                np.array(self.sample_weights)[train_idx] 
                if self.sample_weights is not None 
                else None
            )

            clf = make_pipeline(SVC())
            if self.optimization:
                params = helper.optimize_svm_params(X_train, y_train, clf, self.dataset_name, c.PARAMS_LOSO_PATH)
                clf.set_params(**params)
            clf.fit(X_train, y_train, svc__sample_weight=sample_weight)
            y_pred = clf.predict(X_test)
            actor = self.groups.iloc[test_idx[0]]
            
            #aggiungi le metriche al report
            metrics_handler.add_actor_metrics(y_test, y_pred, actor)
        
        metrics_handler.print_big_report()

    def decision_tree_classifier(self):
        """Classify using Decision Tree"""
        from sklearn import tree
        from sklearn import decomposition
        from sklearn.pipeline import Pipeline

        features = self.features
        loso = LeaveOneGroupOut()
        scaler = StandardScaler()
        columns = features.columns
        features = scaler.fit_transform(features)
        features = pd.DataFrame(features, columns=columns)

        #inizializza le liste per le metriche
        n_classes = len(np.unique(self.target))
        metrics_handler = MetricsHandler(self.dataset_name, 'LOSO_decision_tree', n_classes)

        for train_idx, test_idx in loso.split(features, self.target, groups=self.groups):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = self.target.iloc[train_idx], self.target.iloc[test_idx]
            sample_weight = (
                np.array(self.sample_weights)[train_idx] 
                if self.sample_weights is not None 
                else None
            )

            sc = StandardScaler()
            pca = decomposition.PCA()
            dtreeCLF = tree.DecisionTreeClassifier()
            clf = Pipeline(steps=[('sc', sc), ('pca', pca), ('dtreeCLF', dtreeCLF)])
            if self.optimization:
                params = helper.optimize_decision_tree_params(X_train, y_train, clf, self.dataset_name, c.PARAMS_LOSO_PATH)
                clf.set_params(**params)
            clf.fit(X_train, y_train, dtreeCLF__sample_weight=sample_weight)
            y_pred = clf.predict(X_test)
            actor = self.groups.iloc[test_idx[0]]

            #aggiungi le metriche al report
            metrics_handler.add_actor_metrics(y_test, y_pred, actor)
        
        metrics_handler.print_big_report()

    def lda_classifier(self):
        """Classify using Linear Discriminant Analysis"""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        features = self.features
        loso = LeaveOneGroupOut()
        scaler = StandardScaler()
        columns = features.columns
        features = scaler.fit_transform(features)
        features = pd.DataFrame(features, columns=columns)

        #inizializza le liste per le metriche
        n_classes = len(np.unique(self.target))
        metrics_handler = MetricsHandler(self.dataset_name, 'LOSO_lda', n_classes)

        for train_idx, test_idx in loso.split(features, self.target, groups=self.groups):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = self.target.iloc[train_idx], self.target.iloc[test_idx]

            clf = make_pipeline(LinearDiscriminantAnalysis())
            if self.optimization:
                params = helper.optimize_lda_params(X_train, y_train, clf, self.dataset_name, c.PARAMS_LOSO_PATH)
                clf.set_params(**params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            actor = self.groups.iloc[test_idx[0]]

            #aggiungi le metriche al report
            metrics_handler.add_actor_metrics(y_test, y_pred, actor)
        
        metrics_handler.print_big_report()