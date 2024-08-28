from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import os
import costants as c
import json

def optimize_svm_params(X_train, y_train, clf, dataset_name):
    """Optimize SVM parameters"""
    if not os.path.exists(c.PARAMS_BASE_PATH+dataset_name+'_svm_params.json'):
        print('finding SVM params of '+dataset_name)
        param_grid = {
        'svc__C': [0.1, 0.5, 1, 10, 50],
        'svc__gamma': ['scale', 'auto'],
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
        
        grid = GridSearchCV(clf, param_grid, refit=True, n_jobs=-1)
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
 

    