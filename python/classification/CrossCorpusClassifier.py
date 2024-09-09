import costants as c
from sklearn import svm
import pandas as pd
from classification import helper

class CrossCorpusClassifier:
    def __init__(self, training_corpus,training_labels, testing_corpus, testing_labels, dataset_name):
        self.training_corpus = training_corpus
        self.training_labels = training_labels
        self.testing_corpus = testing_corpus
        self.testing_labels = testing_labels
        self.dataset_name = dataset_name
    
    def __write_report(self, report, dataset_name, classifier_name, latex=False):
        """Write classification report to file, the report needs to be a dictionary"""
        import os
        os.makedirs(c.REPORTS_CROSS_CORPUS_PATH, exist_ok=True)
        report_df =  pd.DataFrame(report).transpose()
        report_df.to_csv(c.REPORTS_CROSS_CORPUS_PATH + dataset_name + '_' + classifier_name + '_report.csv')

        if latex:
            report_df.to_latex(c.REPORTS_CROSS_CORPUS_PATH + dataset_name + '_' + classifier_name + '_report.tex')
    
    def __print_report(self, y_test, y_pred, classifier_name):
        from sklearn.metrics import classification_report, confusion_matrix
        report =  classification_report(y_test, y_pred, labels=[1, 3, 4, 5], target_names=['neu', 'happy', 'sad', 'ang'], zero_division=0)
        confusion = confusion_matrix(y_test, y_pred, labels=[1, 3, 4, 5], normalize=c.NORMALIZE_MATRIX)

        print(report)
        print()
        print(confusion)
        print()
        report_dict = classification_report(y_test, y_pred, labels=[1, 3, 4, 5], 
                                            target_names=['neu', 'happy', 'sad', 'ang'], 
                                            output_dict=True, zero_division=0)
        
        self.__write_report(report_dict, self.dataset_name, classifier_name, latex=True)
        helper.write_cool_confusion_matrix(confusion,['neu', 'happy', 'sad', 'ang'], 
                                           self.dataset_name, classifier_name)
    
    def fit_scaler(self, training_corpus):
        """Fit the scaler"""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(training_corpus)
        return scaler
    
    def get_split(self, scaler):
        columns = self.training_corpus.columns
        X_train = scaler.transform(self.training_corpus)
        X_train = pd.DataFrame(X_train, columns=columns)
        X_test = scaler.transform(self.testing_corpus)
        X_test = pd.DataFrame(X_test, columns=columns)
        return X_train, X_test
    
    def get_best_classifier(self, clf1, clf2, X_test, y_test):
        score1 = clf1.score(X_test, y_test)
        score2 = clf2.score(X_test, y_test)
        if score1 > score2:
            return clf1
        else:
            return clf2
        
    def svm_classifier(self):
        """Classify using the SVM classifier"""
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        #scale the data
        scaler = self.fit_scaler(self.training_corpus)
        X_train, X_test = self.get_split(scaler)
        
        #get labels
        y_train = self.training_labels
        y_test = self.testing_labels

        #initialize the classifier 
        # params_base = helper.load_params(c.PARAMS_BASE_PATH + self.dataset_name + '_svm_params.json')
        # clf1 = make_pipeline(SVC())
        # clf1.set_params(**params_base)
        
        # params_loso = helper.load_params(c.PARAMS_LOSO_PATH + self.dataset_name + '_svm_params.json')
        # clf2 = make_pipeline(SVC())
        # clf2.set_params(**params_loso)

        # #fit the classifier
        # clf1.fit(X_train, y_train)
        # clf2.fit(X_train, y_train)

        # #find the best classifier
        # best_clf = self.get_best_classifier(clf1, clf2, X_test, y_test)
        # if(best_clf == clf1):
        #     print('Base classifier params are better')
        # else:
        #     print('LOSO classifier params are better')
        # y_pred = best_clf.predict(X_test)

        classifier = svm.SVC()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        print(set(y_test) - set(y_pred))
        #write the report
        self.__print_report(y_test, y_pred, 'cross_svm')
    
    def decision_tree_classifier(self):
        """Classify using the Decision Tree classifier"""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.pipeline import make_pipeline
        #scale the data
        scaler = self.fit_scaler(self.training_corpus)
        X_train, X_test = self.get_split(scaler)
        
        #get labels
        y_train = self.training_labels
        y_test = self.testing_labels

        #initialize the classifier 
        # params_base = helper.load_params(c.PARAMS_BASE_PATH + self.dataset_name + '_decision_tree_params.json')
        # clf1 = make_pipeline(DecisionTreeClassifier())
        # clf1.set_params(**params_base)
        
        # params_loso = helper.load_params(c.PARAMS_LOSO_PATH + self.dataset_name + '_decision_tree_params.json')
        # clf2 = make_pipeline(DecisionTreeClassifier())
        # clf2.set_params(**params_loso)

        # #fit the classifier
        # clf1.fit(X_train, y_train)
        # clf2.fit(X_train, y_train)

        # #find the best classifier
        # best_clf = self.get_best_classifier(clf1, clf2, X_test, y_test)
        # if(best_clf == clf1):
        #     print('Base classifier params are better')
        # else:
        #     print('LOSO classifier params are better')
        # y_pred = best_clf.predict(X_test)

        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        #write the report
        self.__print_report(y_test, y_pred, 'cross_decision_tree')
    
    def lda_classifier(self):
        """Classify using the Linear Discriminant Analysis classifier"""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        #scale the data
        scaler = self.fit_scaler(self.training_corpus)
        X_train, X_test = self.get_split(scaler)
        
        #get labels
        y_train = self.training_labels
        y_test = self.testing_labels

        #initialize the classifier 
        # params_base = helper.load_params(c.PARAMS_BASE_PATH + self.dataset_name + '_lda_params.json')
        # clf1 = make_pipeline(LinearDiscriminantAnalysis())
        # clf1.set_params(**params_base)
        
        # params_loso = helper.load_params(c.PARAMS_LOSO_PATH + self.dataset_name + '_lda_params.json')
        # clf2 = make_pipeline(LinearDiscriminantAnalysis())
        # clf2.set_params(**params_loso)

        # #fit the classifier
        # clf1.fit(X_train, y_train)
        # clf2.fit(X_train, y_train)

        # #find the best classifier
        # best_clf = self.get_best_classifier(clf1, clf2, X_test, y_test)
        # if(best_clf == clf1):
        #     print('Base classifier params are better')
        # else:
        #     print('LOSO classifier params are better')
        # y_pred = best_clf.predict(X_test)

        classifier = LinearDiscriminantAnalysis()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        #write the report
        self.__print_report(y_test, y_pred, 'cross_lda')

