import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import sys
sys.path.append(r'')
from basic.function import q_fnc_auc, q_fnc_prc, q_fnc_report, q_fnc_dataset, q_fnc_cv_acc, q_fnc_model

X_train, X_test, y_train, y_test = q_fnc_dataset(4)  #HHH

model = q_fnc_model(0)[3]

kernel = ['linear', 'poly', 'rbf', 'sigmoid']
C = range(1, 30)
gamma = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]

param_dist = {'kernel': kernel,
              'C': C,
              'gamma': gamma}
rs = RandomizedSearchCV(model, param_dist, n_iter=100, cv=5, verbose=1, n_jobs=-1, random_state=0)
rs.fit(X_train, y_train)

arr = []
for key in rs.best_params_:
    arr.append('%s=%s'%(key, rs.best_params_[key]))
print(arr)