from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from classification import helper
from sklearn.preprocessing import StandardScaler

class Base_Classifier:
    def __init__(self, features, target):
        self.features = features
        self.target = target
    
    def svm_classifier(self):
        """Classify using SVM"""
        from sklearn.svm import SVC

        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        clf = make_pipeline(StandardScaler(), SVC())
        params = helper.optimize_svm_params(X_train, y_train, clf)
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        print(report)
        print('\n')
        print(confusion)
