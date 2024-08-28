from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import os
import costants as c
import json
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def write_report(report, dataset_name, classifier_name, latex=False):
    """Write classification report to file, the report needs to be a dictionary"""
    os.makedirs(c.REPORTS_BASE_PATH, exist_ok=True)
    report_df =  pd.DataFrame(report).transpose()
    report_df.to_csv(c.REPORTS_BASE_PATH + dataset_name + '_' + classifier_name + '_report.csv')

    if latex:
        report_df.to_latex(c.REPORTS_BASE_PATH + dataset_name + '_' + classifier_name + '_report.tex')

def write_cool_confusion_matrix(cm, disp_labels, dataset_name, classifier_name):
    """Write confusion matrix to file"""
    os.makedirs(c.REPORTS_BASE_PATH, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    disp.plot()
    plt.savefig(c.REPORTS_BASE_PATH + dataset_name + '_' + classifier_name + '_confusion_matrix.png')
        

def optimize_svm_params(X_train, y_train, clf, dataset_name):
    """Optimize SVM parameters"""
    if not os.path.exists(c.PARAMS_BASE_PATH+dataset_name+'_svm_params.json'):
        print('finding SVM params of '+dataset_name)
        param_grid = {
        'svc__C': [0.1, 0.5, 1, 10, 50],
        'svc__gamma': ['scale', 'auto'],
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
        
        grid = GridSearchCV(clf, param_grid, refit=True, cv=10, n_jobs=-1)
        grid.fit(X_train, y_train)
        print(grid.best_params_)
        os.makedirs(c.PARAMS_BASE_PATH, exist_ok=True)

        with open(c.PARAMS_BASE_PATH + dataset_name+'_svm_params.json', 'w') as f:
            json.dump(grid.best_params_, f)

        return grid.best_params_
    else:
        with open(c.PARAMS_BASE_PATH + dataset_name+'_svm_params.json', 'r') as f:
            print('Loading SVM params of '+dataset_name)
            return json.load(f)
 

    