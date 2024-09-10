from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import decomposition, tree
from sklearn.pipeline import make_pipeline, Pipeline
from classification import helper
from sklearn.preprocessing import StandardScaler
import costants as c
import os
import pandas as pd
class BaseClassifier:
    def __init__(self, features, target, dataset_name):
        self.features = features
        self.target = target
        self.dataset_name = dataset_name

    def __print_report(self,y_test, y_pred, classifier_name):
        report = classification_report(y_test, y_pred, labels=[1, 3, 4, 5], target_names=['neu', 'happy', 'sad', 'ang'])
        confusion = confusion_matrix(y_test, y_pred, labels=[1, 3, 4, 5], normalize=c.NORMALIZE_MATRIX)

        print(report)
        print()
        print(confusion)
        print()
        report_dict = classification_report(y_test, y_pred, labels=[1, 3, 4, 5], target_names=['neu', 'happy', 'sad', 'ang'], output_dict=True)
        self.__write_report(report_dict, self.dataset_name, classifier_name, latex=True)
        helper.write_cool_confusion_matrix(confusion,['neu', 'happy', 'sad', 'ang'], self.dataset_name, classifier_name)

    def __write_report(self, report, dataset_name, classifier_name, latex=False):
        """Write classification report to file, the report needs to be a dictionary"""
        os.makedirs(c.REPORTS_BASE_PATH, exist_ok=True)
        report_df =  pd.DataFrame(report).transpose()
        report_df.to_csv(c.REPORTS_BASE_PATH + dataset_name + '_' + classifier_name + '_report.csv')

        if latex:
            report_df.to_latex(c.REPORTS_BASE_PATH + dataset_name + '_' + classifier_name + '_report.tex')
    
    def svm_classifier(self):
        """Classify using SVM"""
        from sklearn.svm import SVC

        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=c.RANDOM_STATE)
        clf = make_pipeline(StandardScaler(), SVC())
        params = helper.optimize_svm_params(X_train, y_train, clf, self.dataset_name)
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        self.__print_report(y_test, y_pred, 'base_svm')
    
    def decision_tree_classifier(self):
        """Classify using Decision Tree"""
        from sklearn.tree import DecisionTreeClassifier

        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=c.RANDOM_STATE)
        sc = StandardScaler()
        pca = decomposition.PCA()
        dtreeCLF = tree.DecisionTreeClassifier()
        clf = Pipeline(steps=[('sc', sc), ('pca', pca), ('dtreeCLF', dtreeCLF)])
        params = helper.optimize_decision_tree_params(X_train, y_train, clf, self.dataset_name)
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        self.__print_report(y_test, y_pred, 'base_decision_tree')

    def lda_classifier(self):
        """Classify using Linear Discriminant Analysis"""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=c.RANDOM_STATE)
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        params = helper.optimize_lda_params(X_train, y_train, clf, self.dataset_name)
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        self.__print_report(y_test, y_pred, 'base_lda')