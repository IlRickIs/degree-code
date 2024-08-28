from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from classification import helper
from sklearn.preprocessing import StandardScaler

class Base_Classifier:
    def __init__(self, features, target, dataset_name):
        self.features = features
        self.target = target
        self.dataset_name = dataset_name

    def __print_report(self,y_test, y_pred, classifier_name):
        report = classification_report(y_test, y_pred, labels=[1, 3, 4, 5], target_names=['neu', 'happy', 'sad', 'ang'])
        confusion = confusion_matrix(y_test, y_pred, labels=[1, 3, 4, 5], normalize='true')

        print(report)
        print()
        print(confusion)
        print()
        report_dict = classification_report(y_test, y_pred, labels=[1, 3, 4, 5], target_names=['neu', 'happy', 'sad', 'ang'], output_dict=True)
        helper.write_report(report_dict, self.dataset_name, classifier_name, latex=True)
        helper.write_cool_confusion_matrix(confusion,['neu', 'happy', 'sad', 'ang'], self.dataset_name, classifier_name)
    
    def svm_classifier(self):
        """Classify using SVM"""
        from sklearn.svm import SVC

        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        clf = make_pipeline(StandardScaler(), SVC())
        params = helper.optimize_svm_params(X_train, y_train, clf, self.dataset_name)
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        self.__print_report(y_test, y_pred, '_base_svm')
    
    def decision_tree_classifier(self):
        """Classify using Decision Tree"""
        from sklearn.tree import DecisionTreeClassifier

        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier()
        params = helper.optimize_decision_tree_params(X_train, y_train, clf, self.dataset_name)
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        self.__print_report(y_test, y_pred, '_base_decision_tree')

    def lda_classifier(self):
        """Classify using Linear Discriminant Analysis"""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        params = helper.optimize_lda_params(X_train, y_train, clf, self.dataset_name)
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        self.__print_report(y_test, y_pred, '_base_lda')