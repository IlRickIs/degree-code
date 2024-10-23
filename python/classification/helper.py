from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import os
import costants as c
import json
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def write_cool_confusion_matrix(cm, disp_labels, dataset_name, classifier_name):
    """Write confusion matrix to file"""
    os.makedirs(c.REPORTS_CM_PATH, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    fig, ax = plt.subplots()
    cax = ax.imshow(cm, cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(cax)
    disp.plot(ax=ax, cmap='viridis', colorbar=False)
    disp.ax_.set_title(dataset_name + '_' + classifier_name + '_confusion_matrix')
    plt.savefig(c.REPORTS_CM_PATH + dataset_name + '_' + classifier_name + '_confusion_matrix.png')
    plt.close()
        

def optimize_svm_params(X_train, y_train, clf, dataset_name, params_path = c.PARAMS_BASE_PATH):
    """Optimize SVM parameters"""
    if not os.path.exists(params_path+dataset_name+'_svm_params.json'):
        #print('finding SVM params of '+dataset_name)
        param_grid = {
        'svc__C': [0.1, 0.5, 1, 10],
        'svc__gamma': ['scale', 'auto'],
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
        
        grid = GridSearchCV(clf, param_grid, refit=True, cv=10, n_jobs=-1)
        grid.fit(X_train, y_train)
        print(grid.best_params_)
        os.makedirs(params_path, exist_ok=True)

        with open(params_path + dataset_name+'_svm_params.json', 'w') as f:
            json.dump(grid.best_params_, f)

        return grid.best_params_
    else:
        with open(params_path + dataset_name+'_svm_params.json', 'r') as f:
            #print('Loading SVM params of '+dataset_name)
            return json.load(f)

def optimize_decision_tree_params(X_train, y_train, clf, dataset_name, params_path = c.PARAMS_BASE_PATH):
    """Optimize Decision Tree parameters"""
    if not os.path.exists(params_path+dataset_name+'_decision_tree_params.json'):
        n_components = list(range(1, X_train.shape[1]+1))
        criterion = ['gini', 'entropy']
        max_depth = [2, 4, 6, 8, 10, 15, 30, 35, None]

        parameters = dict(pca__n_components=n_components,
                          dtreeCLF__criterion=criterion,
                          dtreeCLF__max_depth=max_depth
                        )
        
        grid = GridSearchCV(clf, parameters, refit=True, cv=10, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        print(grid.best_params_)
        os.makedirs(params_path, exist_ok=True)

        with open(params_path + dataset_name+'_decision_tree_params.json', 'w') as f:
            json.dump(grid.best_params_, f)

        return grid.best_params_
    else:
        with open(params_path + dataset_name+'_decision_tree_params.json', 'r') as f:
            return json.load(f)
    
def optimize_lda_params(X_train, y_train, clf, dataset_name, params_path = c.PARAMS_BASE_PATH):
    """Optimize LDA parameters"""
    if not os.path.exists(params_path+dataset_name+'_lda_params.json'):
        param_grid = {
            'lineardiscriminantanalysis__solver': ['lsqr', 'eigen'],
            'lineardiscriminantanalysis__shrinkage': ['auto', None]}
        
        grid = GridSearchCV(clf, param_grid, refit=True, cv=10, n_jobs=-1)
        grid.fit(X_train, y_train)
        print(grid.best_params_)
        os.makedirs(params_path, exist_ok=True)

        with open(params_path + dataset_name+'_lda_params.json', 'w') as f:
            json.dump(grid.best_params_, f)

        return grid.best_params_
    else:
        with open(params_path + dataset_name+'_lda_params.json', 'r') as f:
            return json.load(f)
    
def load_params(path):
    """Load parameters from file"""
    with open(path, 'r') as f:
        return json.load(f)
 

    